import gdown


def download_pretrained_model():
    url = 'https://drive.google.com/file/d/1ykOQ7JxwCGWtSqz0NPfuUdvGL5Mawc3C/view?usp=sharing'
    output_path = './pretrained_model.ckpt'
    gdown.download(url, output_path, quiet=False, fuzzy=True)


if __name__ == '__main__':
    download_pretrained_model()
