import os
import csv

def classify_abs(val):
    aval = abs(val)
    if aval == 0:
        return 1
    elif aval == 1:
        return 2
    elif 2 <= aval <= 7:
        return 3
    else:
        return 4

def classify_sign(val):
    if -1 <= val <= 1:
        return 1
    elif val >= 2:
        return 2
    elif val <= -2:
        return 3

    elif 1 < val < 2:
        return 2
    elif -2 < val < -1:
        return 3

    return 1

def process_csv(input_csv, output_csv):
    with open(input_csv, 'r', encoding='utf-8') as fin, \
         open(output_csv, 'w', encoding='utf-8', newline='') as fout:

        reader = csv.reader(fin)
        writer = csv.writer(fout)


        header = next(reader)

        new_header = header + ["dx", "dy", "abs_dx_cat", "abs_dy_cat","dx_sign_cat","dy_sign_cat"]
        writer.writerow(new_header)

        last_uid = None
        last_x = 0
        last_y = 0

        for row in reader:
            uid_str, d_str, t_str, x_str, y_str = row
            uid = int(uid_str)
            x = int(x_str)
            y = int(y_str)

            if uid != last_uid:
                dx = 0
                dy = 0
                last_uid = uid
            else:
                dx = x - last_x
                dy = y - last_y
            last_x = x
            last_y = y

            abs_dx_cat = classify_abs(dx)
            abs_dy_cat = classify_abs(dy)
            dx_sign_cat = classify_sign(dx)
            dy_sign_cat = classify_sign(dy)

            new_row = row + [
                str(dx),
                str(dy),
                str(abs_dx_cat),
                str(abs_dy_cat),
                str(dx_sign_cat),
                str(dy_sign_cat)
            ]
            writer.writerow(new_row)

def main():
    base_path = os.getcwd()
    input_csv = os.path.join(base_path, 'data/cityA-dataset.csv')
    output_csv = os.path.join(base_path, 'data/cityA-dataset-with-dxdy.csv')

    process_csv(input_csv, output_csv)
    print("Done. Result saved to", output_csv)

if __name__ == "__main__":
    main()
