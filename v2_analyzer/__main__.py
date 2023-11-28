import argparse

from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

from get_menu.get_menu.spiders.menu_spider import MenuSpider


def make_parser():
    parser = argparse.ArgumentParser(
        description='analyzerの動作方法を指定する'
    )
    parser.add_argument('runtype', help='getdata, 最新データをwebから取得する(viewの前に実行)\n'
                                        'view, DBに登録したデータを可視化する')
    args = parser.parse_args()
    return args


def run_spider():
    settings = get_project_settings()
    process = CrawlerProcess(settings)
    process.crawl(MenuSpider)
    process.start()


if __name__ == '__main__':
    args = make_parser()
    if args.runtype == 'getdata':
        run_spider()
    elif args.runtype == 'view':
        raise Exception('可視化は未実装')
    else:
        raise ValueError('getdata か view を指定してください')
