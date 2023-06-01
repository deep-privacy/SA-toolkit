import argparse
import os


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--mls_root', type=str, help="Path to the root of mls dataset")
    parser.add_argument('--data_split', type=str, help="Data split to process (train, dev or test)")
    parser.add_argument('--out_dir', type=str, help="Output directory for mls as kaldi dataset")
    parser.add_argument('--use_book_in_spk', type=bool, default=True, help="Indicates if speaker id contains book id or not")

    args = parser.parse_args()
    mls_root = args.mls_root
    data_split = args.data_split
    out_dir = args.out_dir
    use_book_in_spk = args.use_book_in_spk

    # Create output directory if not exists
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Load spk2gender data for all speakers (in train, dev and test)
    all_spk2gender_dict = {}
    with open(os.path.join(mls_root, "metainfo.txt"), 'r') as metainfo_file:
        # Skip header
        header = metainfo_file.readline()
        # Load gender information
        for line in metainfo_file:
            split_line = line.split("|")
            spk = split_line[0].strip()
            book = split_line[4].strip()
            gender = split_line[1].strip().lower()
            if use_book_in_spk:
                spk_id = spk + "-" + book
            else:
                spk_id = spk
            all_spk2gender_dict[spk_id] = gender


    data_path = os.path.join(mls_root, data_split)
    spk2utt_dict = {}
    utt2spk_list = []
    wav_scp_list = []
    spk2gender_dict = {}
    # Browse audio directory to extract information about each utterance and speaker
    audio_path = os.path.join(data_path, "audio")
    for spk in os.listdir(audio_path):
        for book in os.listdir(os.path.join(audio_path, spk)):
            if use_book_in_spk:
                spk_id = spk + "-" + book
                spk_id_utt = spk_id
            else:
                spk_id = spk
                spk_id_utt = ""

            # Add current speaker to spk2gender dict only the first time we encounter the current speaker
            spk2gender_dict.setdefault(spk_id, all_spk2gender_dict[spk_id])

            # Browse files
            curRoot = os.path.join(audio_path, spk, book)
            for file in os.listdir(curRoot):
                if os.path.splitext(file)[1] == ".flac":
                    utt = spk_id_utt + "_" + os.path.splitext(file)[0]
                    # Add command for on-the-fly conversion flac file in kaldi
                    wav_scp_list.append(utt + " flac -c -d -s " + os.path.join(curRoot, file) + " |")
                    spk2utt_dict.setdefault(spk_id, []).append(utt)
                    utt2spk_list.append(utt + " " + spk_id)


    # Create wav.scp file
    write_list_to_file(wav_scp_list, os.path.join(out_dir, "wav.scp"))

    # Create spk2gender file
    spk2gender_list = [spk_id + " " + gender for spk_id, gender in spk2gender_dict.items()]
    write_list_to_file(spk2gender_list, os.path.join(out_dir, "spk2gender"))

    # Create spk2utt file
    spk2utt_list = [spk + " " + " ".join(utt_list) for spk, utt_list in spk2utt_dict.items()]
    write_list_to_file(spk2utt_list, os.path.join(out_dir, "spk2utt"))

    # Create utt2spk file
    write_list_to_file(utt2spk_list, os.path.join(out_dir, "utt2spk"))

    # Create text file
    text_list = []
    for line in open(os.path.join(data_path, "transcripts.txt"), 'r'):
        split_line = line.split()
        spk = split_line[0].split("_")[0]
        book = split_line[0].split("_")[1]
        if use_book_in_spk:
            spk_id = spk + "-" + book + "_"
        else:
            spk_id = ""
        text_list.append(spk_id + split_line[0] + " " + " ".join(split_line[1:]))
    write_list_to_file(text_list, os.path.join(out_dir, "text"))

    # Create utt2dur file
    duration_list = []
    for line in open(os.path.join(data_path, "segments.txt"), 'r'):
        split_line = line.split()
        spk = split_line[0].split("_")[0]
        book = split_line[0].split("_")[1]
        if use_book_in_spk:
            spk_id = spk + "-" + book + "_"
        else:
            spk_id = ""
        duration_list.append(spk_id + split_line[0] + " " + str(round(float(split_line[3]) - float(split_line[2]), 2)))
    write_list_to_file(duration_list, os.path.join(out_dir, "utt2dur"))


def write_list_to_file(list_to_write, filepath):
    # Write data list to a file compatible with kaldi
    out_file = open(filepath, "w")
    out_file.write("\n".join(list_to_write))
    out_file.write("\n")
    out_file.close()

if __name__ == "__main__":
    main()
