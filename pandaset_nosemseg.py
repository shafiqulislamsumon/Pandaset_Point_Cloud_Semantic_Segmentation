import os
import shutil


def remove_sequence_folder():
    dir = './dataset/Pandaset/'
    no_sem_seg = [4, 6, 8, 12, 14, 18, 20, 45, 47, 48, 50, 51, 55, 59, 62, 63, 68, 74, 79, 85, 86, 91, 92, 93, 99, 100, 104]
    for seq in no_sem_seg:
        path = dir + str(seq).zfill(3)
        print("path : ", path)
        shutil.rmtree(path, ignore_errors=True)


def pandaset_info():
    dir = './dataset/Pandaset/'
    seq_list = os.listdir(dir)
    seq_list = sorted(seq_list)
    print("##### seq len : ", len(seq_list))
    print("##### seq list : ", seq_list)


if __name__ == '__main__':
    #remove_sequence_folder()
    pandaset_info()
