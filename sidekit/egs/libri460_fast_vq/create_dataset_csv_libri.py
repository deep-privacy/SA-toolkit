# -*- coding: utf8 -*-
import csv
import os
import torchaudio

def main():
    if not os.path.exists("list"):
        os.makedirs("list")
    out_filname = "list/libri360.csv"
    libri_root = "/lium/raid01_b/pchampi/lab/asr-based-privacy-preserving-separation/pkwrap/egs/librispeech/v1/corpora/LibriSpeech/"

    # Retrieve gender for Librispeech speakers
    spk_file = open(os.path.join(libri_root, "SPEAKERS.TXT"), "r")
    spk_gender_dict = {}
    for line in spk_file:
        if line[0] != ";":
            split_line = line.split("|")
            spk_gender_dict[split_line[0].strip()] = split_line[1].strip().lower()

    spk_list = []
    # Browse directories to retrieve list audio files
    with open(out_filname, 'w', newline='') as out_csv_file:
        csv_writer = csv.writer(out_csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        # Write header
        csv_writer.writerow(["speaker_idx", "database", "speaker_id", "start", "duration", "file_id", "gender"])

        for root, dirs, files in os.walk(libri_root):
            for file in files:
                if file.split(".")[-1] == "flac":
                    dataset = root.split("/")[-3]
                    if dataset not in ['train-clean-100', 'train-clean-360']:
                       continue
                    spk_id = file.split("-")[0]
                    if spk_id not in spk_list:
                        spk_list.append(spk_id)
                        print("spk count : ", len(spk_list))
                    spk_idx = spk_list.index(spk_id)
                    start = 0
                    file_path = os.path.join(root.replace(libri_root, ""), file)
                    audio_info = torchaudio.info(os.path.join(root, file))
                    duration = audio_info.num_frames / audio_info.sample_rate
                    file_id = file_path.split(".")[0] # Remove file extension
                    gender = spk_gender_dict[spk_id]

                    csv_writer.writerow([spk_idx, dataset, spk_id, start, duration, file_id, gender])

if __name__ == "__main__":
    main()
