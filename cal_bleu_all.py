import os
import json
import geobleu
import statistics

def analyze_file(json_path, output_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    generated_list = data["generated"]
    reference_list = data["reference"]

    geobleu_vals = []
    dtw_vals = []

    with open(output_path, 'w', encoding='utf-8') as out:
        for i in range(len(generated_list)):
            generated = generated_list[i]
            reference = reference_list[i]

            geobleu_val = geobleu.calc_geobleu(generated, reference, processes=3)
            dtw_val = geobleu.calc_dtw(generated, reference, processes=3)

            geobleu_vals.append(geobleu_val)
            dtw_vals.append(dtw_val)

            out.write(f"轨迹{i+1}: geobleu = {geobleu_val:.4f}, dtw = {dtw_val:.4f}\n")

        out.write("\n=== geobleu stat ===\n")
        out.write(f"avg: {statistics.mean(geobleu_vals):.4f}\n")
        out.write(f"max: {max(geobleu_vals):.4f}\n")
        out.write(f"min: {min(geobleu_vals):.4f}\n")

        out.write("\n=== dtw stat ===\n")
        out.write(f"avg: {statistics.mean(dtw_vals):.4f}\n")
        out.write(f"max: {max(dtw_vals):.4f}\n")
        out.write(f"min: {min(dtw_vals):.4f}\n")

    print(f"[done]: {json_path} -> {output_path}")

def main():
    folder = "/workspace/LP-BERT_New"
    output_dir = os.path.join(folder, "eval_results")
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(folder):
        if file.endswith(".json"):
            json_path = os.path.join(folder, file)
            output_path = os.path.join(output_dir, f"{os.path.splitext(file)[0]}_results.txt")
            analyze_file(json_path, output_path)

if __name__ == "__main__":
    main()
