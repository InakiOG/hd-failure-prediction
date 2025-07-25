import subprocess
import json
from datetime import datetime
import csv # pip install csv
# smartmontools needed: choco install smartmontools
import time
import tempfile
import argparse

def add_to_csv(json_file, csv_file):
    smart_numbers = []
    with open(json_file, 'r') as f:
        data = json.load(f)
    if 'smart_support' not in data:
        return None
    if 'local_time' not in data or 'serial_number' not in data or 'model_name' not in data or 'ata_smart_attributes' not in data:
        return None

    date = data['local_time']['asctime']
    serial_number = data['serial_number']
    model = data['model_name']
    capacity_bytes = None
    failure = None
    datacenter = None
    cluster_id = None
    vault_id = None
    pod_id = None
    pod_slot_num = None
    is_legacy_format = None
    smart = data['ata_smart_attributes']['table']
    new_row = [date, serial_number, model, capacity_bytes, failure,
               datacenter, cluster_id, vault_id, pod_id, pod_slot_num, is_legacy_format]

    first_row = ['date','serial_number','model','capacity_bytes','failure','datacenter','cluster_id','vault_id','pod_id','pod_slot_num','is_legacy_format','smart_1_normalized','smart_1_raw','smart_2_normalized','smart_2_raw','smart_3_normalized','smart_3_raw','smart_4_normalized','smart_4_raw','smart_5_normalized','smart_5_raw','smart_7_normalized','smart_7_raw','smart_8_normalized','smart_8_raw','smart_9_normalized','smart_9_raw','smart_10_normalized','smart_10_raw','smart_11_normalized','smart_11_raw','smart_12_normalized','smart_12_raw','smart_13_normalized','smart_13_raw','smart_15_normalized','smart_15_raw','smart_16_normalized','smart_16_raw','smart_17_normalized','smart_17_raw','smart_18_normalized','smart_18_raw','smart_22_normalized','smart_22_raw','smart_23_normalized','smart_23_raw','smart_24_normalized','smart_24_raw','smart_27_normalized','smart_27_raw','smart_71_normalized','smart_71_raw','smart_82_normalized','smart_82_raw','smart_90_normalized','smart_90_raw','smart_160_normalized','smart_160_raw','smart_161_normalized','smart_161_raw','smart_163_normalized','smart_163_raw','smart_164_normalized','smart_164_raw','smart_165_normalized','smart_165_raw','smart_166_normalized','smart_166_raw','smart_167_normalized','smart_167_raw','smart_168_normalized','smart_168_raw','smart_169_normalized','smart_169_raw','smart_170_normalized','smart_170_raw','smart_171_normalized','smart_171_raw','smart_172_normalized','smart_172_raw','smart_173_normalized','smart_173_raw','smart_174_normalized','smart_174_raw','smart_175_normalized','smart_175_raw','smart_176_normalized','smart_176_raw','smart_177_normalized','smart_177_raw','smart_178_normalized','smart_178_raw','smart_179_normalized','smart_179_raw','smart_180_normalized','smart_180_raw','smart_181_normalized','smart_181_raw','smart_182_normalized','smart_182_raw','smart_183_normalized','smart_183_raw','smart_184_normalized','smart_184_raw','smart_187_normalized','smart_187_raw','smart_188_normalized','smart_188_raw','smart_189_normalized','smart_189_raw','smart_190_normalized','smart_190_raw','smart_191_normalized','smart_191_raw','smart_192_normalized','smart_192_raw','smart_193_normalized','smart_193_raw','smart_194_normalized','smart_194_raw','smart_195_normalized','smart_195_raw','smart_196_normalized','smart_196_raw','smart_197_normalized','smart_197_raw','smart_198_normalized','smart_198_raw','smart_199_normalized','smart_199_raw','smart_200_normalized','smart_200_raw','smart_201_normalized','smart_201_raw','smart_202_normalized','smart_202_raw','smart_206_normalized','smart_206_raw','smart_210_normalized','smart_210_raw','smart_218_normalized','smart_218_raw','smart_220_normalized','smart_220_raw','smart_222_normalized','smart_222_raw','smart_223_normalized','smart_223_raw','smart_224_normalized','smart_224_raw','smart_225_normalized','smart_225_raw','smart_226_normalized','smart_226_raw','smart_230_normalized','smart_230_raw','smart_231_normalized','smart_231_raw','smart_232_normalized','smart_232_raw','smart_233_normalized','smart_233_raw','smart_234_normalized','smart_234_raw','smart_235_normalized','smart_235_raw','smart_240_normalized','smart_240_raw','smart_241_normalized','smart_241_raw','smart_242_normalized','smart_242_raw','smart_244_normalized','smart_244_raw','smart_245_normalized','smart_245_raw','smart_246_normalized','smart_246_raw','smart_247_normalized','smart_247_raw','smart_248_normalized','smart_248_raw','smart_250_normalized','smart_250_raw','smart_251_normalized','smart_251_raw','smart_252_normalized','smart_252_raw','smart_254_normalized','smart_254_raw','smart_255_normalized','smart_255_raw']
    for attribute in first_row[11:]:
        smart_numbers.append(attribute.split('_')[1])

    smart_values = [None] * len(smart_numbers)
    smart_v = {obj["id"]: [obj["value"], obj['raw']['value']]
               for obj in smart}
    for id, value in smart_v.items():
        if(str(id) not in smart_numbers):
            continue
        smart_values[smart_numbers.index(str(id))] = value[0]
        smart_values[smart_numbers.index(str(id))+1] = value[1]

    row = new_row + smart_values
    return row


command = 'smartctl --scan'
output_path = tempfile.gettempdir() + '\smart.json'
parser = argparse.ArgumentParser(description="Provide a path for the CSV file.")

parser.add_argument('CsvPath', metavar='csvpath', type=str, help='the path to the CSV file')

args = parser.parse_args()


while True:
    csv_path = f"{args.CsvPath}/{datetime.now().strftime('%Y-%m-%d')}.csv"
    print(csv_path)
    print(datetime.now())
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    devices = [x for x in result.stdout.split('\n') if x != '']
    rows = []
    for device in devices:
        d = device.split()[0]
        json_command = f'smartctl -a -j {d} > {output_path}'
        json_result = subprocess.run(json_command, shell=True,
                                        capture_output=True, text=True)
        info = add_to_csv(output_path, csv_path)
        if info is None:
            print(f"No SMART data available for {d}")
        else:
            print(f"Adding data for {d}")
            rows.append(info)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)
    f.close()
    time.sleep(100)