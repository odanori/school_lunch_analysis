import argparse

import analyzer.read_data as read_data


def make_parser():
    parser = argparse.ArgumentParser(
        description='調べたいデータのzipファイル名(.zipは入れない)を引数に指定。指定しない場合はエラー')
    parser.add_argument('target', help='調べたいデータのzipファイル名')
    args = parser.parse_args()
    return args


def run():
    args = make_parser()
    read_data.read_zip_file(args.target)


if __name__ == '__main__':
    run()
