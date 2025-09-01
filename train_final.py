import os, argparse, logging, random, datetime, math
from tqdm import tqdm
import numpy as np
import torch, torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter

from dataset_final import *
from model_final import *


def set_random_seed(seed=0):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False


def collate_fn(batch):
    keys=['d','t','input_x','input_y','time_delta',
          'label_x','label_y','len',
          'dx_sign_cat','dy_sign_cat','dx_cat','dy_cat']
    e={k:[b[k] for b in batch] for k in keys}; pad=lambda l:pad_sequence(l,batch_first=True,padding_value=0)
    return {'d':pad(e['d']), 't':pad(e['t']),
            'input_x':pad(e['input_x']), 'input_y':pad(e['input_y']),
            'time_delta':pad(e['time_delta']),
            'label_x':pad(e['label_x']), 'label_y':pad(e['label_y']),
            'len':torch.tensor(e['len']).squeeze(-1),  
            'dx_sign_cat':pad(e['dx_sign_cat']), 'dy_sign_cat':pad(e['dy_sign_cat']),
            'dx_cat':pad(e['dx_cat']), 'dy_cat':pad(e['dy_cat'])}


class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-3):
        self.best = math.inf; self.patience = patience
        self.delta = min_delta; self.counter = 0
    def step(self, metric):
        if metric < self.best - self.delta:
            self.best = metric; self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def topk_acc(logits,target,k=1):
    return ((logits.topk(k,-1).indices==target.unsqueeze(-1)).any(-1)
            .float().mean().item())


def ade_fde(pred_xy,true_xy):
    dist=torch.norm(pred_xy-true_xy,dim=-1)
    return dist.mean().item(),dist[-1].item()


@torch.no_grad()
def evaluate(model, loader, criterion, criterion_sub1,
             criterion_sub2, device, use_sub1, use_sub2):
    model.eval()
    loss_sum = sub1_sum = sub2_sum = n_tok = 0
    top1_sum = top5_sum = 0
    ade_sum = fde_sum = seq_cnt = 0
    sub1_tok = sub2_tok = 1e-8

    for batch in loader:
        batch = {k:v.to(device) for k,v in batch.items()}
        outputs = model(batch['d'], batch['t'],
                        batch['input_x'], batch['input_y'],
                        batch['time_delta'], batch['len'])
        # main
        pred_main = outputs['main_out']
        label_xy  = torch.stack((batch['label_x'], batch['label_y']), dim=-1)
        mask = (batch['input_x']==201)
        mask = torch.cat((mask.unsqueeze(-1),mask.unsqueeze(-1)),-1)

        logits_sel = pred_main[mask].view(-1,200)
        label_sel  = label_xy[mask].view(-1)
        loss_main  = criterion(logits_sel, label_sel)

        m = label_sel.size(0)
        loss_sum += loss_main.item()*m; n_tok += m
        top1_sum += topk_acc(logits_sel,label_sel,1)*m
        top5_sum += topk_acc(logits_sel,label_sel,5)*m

        # ADE/FDE
        B=pred_main.size(0)
        for b in range(B):
            seq_mask = mask[b,:,0]
            if seq_mask.sum()==0: continue
            logits_xy = pred_main[b][seq_mask]
            preds = torch.stack([                       ### <<< FIX -- vectorised
                     logits_xy[:,0].argmax(-1),
                     logits_xy[:,1].argmax(-1)], dim=-1).float()
            true_xy = label_xy[b][seq_mask].float()
            a,f=ade_fde(preds,true_xy); ade_sum+=a; fde_sum+=f; seq_cnt+=1

        # sub1
        if use_sub1:
            dist_out = outputs['distance_out']
            dist_lab = torch.stack((batch['dx_cat'],batch['dy_cat']),dim=-1)
            sm=(batch['d']<61); sm=torch.cat((sm.unsqueeze(-1),sm.unsqueeze(-1)),-1)
            ls1 = criterion_sub1(dist_out[sm].view(-1,5), dist_lab[sm].view(-1))
            sub1_sum += ls1.item()* dist_lab[sm].numel()
            sub1_tok += dist_lab[sm].numel()
        # sub2
        if use_sub2:
            dir_out = outputs['direction_out']
            dir_lab = torch.stack((batch['dx_sign_cat'],batch['dy_sign_cat']),dim=-1)
            sm=(batch['d']<61); sm=torch.cat((sm.unsqueeze(-1),sm.unsqueeze(-1)),-1)
            ls2 = criterion_sub2(dir_out[sm].view(-1,4), dir_lab[sm].view(-1))
            sub2_sum += ls2.item()* dir_lab[sm].numel()
            sub2_tok += dir_lab[sm].numel()

    val_loss = loss_sum / n_tok
    val_sub1 = sub1_sum / sub1_tok if use_sub1 else 0
    val_sub2 = sub2_sum / sub2_tok if use_sub2 else 0
    top1 = top1_sum / n_tok
    top5 = top5_sum / n_tok
    ade  = ade_sum / seq_cnt
    fde  = fde_sum / seq_cnt
    return val_loss, val_sub1, val_sub2, top1, top5, ade, fde


def task1(args):
    base = os.getcwd()
    data_path = os.path.join(base, 'data/dataset_train.csv')
    data_path1 = os.path.join(base, 'data/dataset_val.csv')
    run = f'bs{args.batch_size}_ep{args.epochs}_emb{args.embed_size}_L{args.layers_num}_H{args.heads_num}_lr{args.lr}'
    logd = os.path.join('log','Final 1',run); tbd = os.path.join('tb_log','Final 1',run); ckptd = os.path.join('checkpoint','Final 1',run)
    for d in (logd,tbd,ckptd): os.makedirs(d,exist_ok=True)

    logging.basicConfig(filename=os.path.join(logd,'train.log'),
                        level=logging.INFO,format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    writer = SummaryWriter(tbd)

    train_ds = HuMobDatasetTrain(data_path)
    val_ds   = HuMobDatasetVal  (data_path1)
    train_dl = DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,
                          collate_fn=collate_fn,num_workers=args.num_workers)
    val_dl   = DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,
                          collate_fn=collate_fn,num_workers=args.num_workers)

    device = torch.device(f'cuda:{args.cuda}')
    model  = HumobTransformer(args.layers_num,args.heads_num,args.embed_size,
                    args.use_subtask1,args.use_subtask2).to(device)
    if torch.cuda.device_count()>1: model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=5e-7)

    criterion      = nn.CrossEntropyLoss()
    criterion_sub1 = nn.CrossEntropyLoss(ignore_index=0)
    criterion_sub2 = nn.CrossEntropyLoss(ignore_index=0)

    early = EarlyStopping(patience=15)
    best_val = math.inf

    for epoch in range(args.epochs):
        model.train()
        epoch_loss_main = epoch_loss_sub1 = epoch_loss_sub2 = 0  

        for batch in tqdm(train_dl, desc=f"Epoch {epoch:02d}"):
            batch = {k:v.to(device) for k,v in batch.items()}
            outputs = model(batch['d'], batch['t'],
                            batch['input_x'], batch['input_y'],
                            batch['time_delta'], batch['len'])
            pred_main = outputs['main_out']
            label_xy  = torch.stack((batch['label_x'], batch['label_y']), dim=-1)

            pred_mask = (batch['input_x'] == 201)
            pred_mask = torch.cat((pred_mask.unsqueeze(-1), pred_mask.unsqueeze(-1)), dim=-1)

            loss_main = criterion(pred_main[pred_mask].view(-1, 200),
                                  label_xy[pred_mask].view(-1))
            total_loss = loss_main

            if args.use_subtask1:
                dist_out = outputs['distance_out']
                dist_label = torch.stack((batch['dx_cat'], batch['dy_cat']), dim=-1)
                sub_mask = (batch['d'] < 61)
                sub_mask = torch.cat((sub_mask.unsqueeze(-1), sub_mask.unsqueeze(-1)), dim=-1)
                loss_sub1 = criterion_sub1(
                    dist_out[sub_mask].view(-1, 5),
                    dist_label[sub_mask].view(-1)
                )
                total_loss = total_loss + loss_sub1*0.5
                epoch_loss_sub1 += loss_sub1.item()
            else:
                loss_sub1 = torch.tensor(0., device=device)

            if args.use_subtask2:
                dir_out = outputs['direction_out']
                dir_label = torch.stack((batch['dx_sign_cat'], batch['dy_sign_cat']), dim=-1)
                sub_mask = (batch['d'] < 61)
                sub_mask = torch.cat((sub_mask.unsqueeze(-1), sub_mask.unsqueeze(-1)), dim=-1)
                loss_sub2 = criterion_sub2(
                    dir_out[sub_mask].view(-1, 4),
                    dir_label[sub_mask].view(-1)
                )
                total_loss = total_loss + loss_sub2*0.8
                epoch_loss_sub2 += loss_sub2.item()
            else:
                loss_sub2 = torch.tensor(0., device=device)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss_main += loss_main.item()


        train_loss_main = epoch_loss_main / len(train_dl)
        writer.add_scalar('loss_main/train', train_loss_main, epoch)
        if args.use_subtask1:
            writer.add_scalar('loss_sub1/train', epoch_loss_sub1 / len(train_dl), epoch)  
        if args.use_subtask2:
            writer.add_scalar('loss_sub2/train', epoch_loss_sub2 / len(train_dl), epoch) 


        val_loss, val_loss_sub1, val_loss_sub2, top1, top5, ade, fde = evaluate(
            model, val_dl, criterion, criterion_sub1, criterion_sub2,
            device, args.use_subtask1, args.use_subtask2)

        writer.add_scalar('loss_main/val', val_loss, epoch)
        if args.use_subtask1:
            writer.add_scalar('loss_sub1/val', val_loss_sub1, epoch)
        if args.use_subtask2:
            writer.add_scalar('loss_sub2/val', val_loss_sub2, epoch)
        writer.add_scalar('acc/top1', top1, epoch)
        writer.add_scalar('acc/top5', top5, epoch)
        writer.add_scalar('metric/ADE', ade, epoch)
        writer.add_scalar('metric/FDE', fde, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)


        logging.info(
            f'epoch:{epoch:03d} '
            f'train_main:{train_loss_main:.3f} '
            f'train_sub1:{epoch_loss_sub1/len(train_dl):.3f} '
            f'train_sub2:{epoch_loss_sub2/len(train_dl):.3f} '
            f'val_main:{val_loss:.3f} '
            f'val_sub1:{val_loss_sub1:.3f} '
            f'val_sub2:{val_loss_sub2:.3f} '
            f'top1:{top1:.3f} top5:{top5:.3f} '
            f'ade:{ade:.2f} fde:{fde:.2f}'
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(ckptd, f'best_{best_val:.3f}.pth'))
        if early.step(val_loss):
            print("Early-Stopping triggered."); break

        scheduler.step()

    torch.save(model.state_dict(), os.path.join(
        ckptd, f'final_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs',     type=int, default=200)
    parser.add_argument('--num_workers',type=int, default=2)
    parser.add_argument('--embed_size', type=int, default=128)
    parser.add_argument('--layers_num', type=int, default=4)
    parser.add_argument('--heads_num',  type=int, default=8)
    parser.add_argument('--cuda',       type=int, default=0)
    parser.add_argument('--lr',         type=float, default=2e-5)
    parser.add_argument('--seed',       type=int, default=0)
    parser.add_argument('--use_subtask1', action='store_true')
    parser.add_argument('--use_subtask2', action='store_true')
    args = parser.parse_args()

    set_random_seed(args.seed)
    task1(args)
