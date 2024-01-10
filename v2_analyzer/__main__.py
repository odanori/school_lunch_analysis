import argparse

from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

from v2_analyzer.get_data_from_db import take_data
from v2_analyzer.get_menu.get_menu.spiders.menu_spider import MenuSpider
from v2_analyzer.graphs.make_graph import output_graph


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


def run_viewer():
    all_data = take_data()
    output_graph(all_data)


# TODO:可視化機能の増強
def run():
    args = make_parser()
    if args.runtype == 'getdata':
        run_spider()
    elif args.runtype == 'view':
        run_viewer()
    else:
        raise ValueError('getdata か view を指定してください')


if __name__ == '__main__':
    run()
