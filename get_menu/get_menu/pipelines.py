# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html

import io
from pathlib import Path

import pandas as pd
import psycopg2
from scrapy.exceptions import DropItem, NotConfigured
from sqlalchemy import create_engine

from get_menu.preparation.preprocessor import data_processor

# useful for handling different item types with a single interface
# from itemadapter import ItemAdapter
# import re


class DataProcess:

    def __init__(self):
        self.download_path = Path('./get_menu/tmp/')
        if not self.download_path.exists():
            self.download_path.mkdir()

    def process_item(self, item, spider):

        base_data = self.prepare_base_data(item)
        preprocessed_data = data_processor(base_data)
        tmp_csv_path = self.download_path / item['filename']
        preprocessed_data.to_csv(tmp_csv_path, encoding='utf-8', index=False)
        item['csv_data'] = tmp_csv_path
        return item

    def read_byte_csv(self, csv_data):
        data = pd.read_csv(io.BytesIO(csv_data), encoding='cp932', index_col=None)
        return data

    def add_info_to_data(self, data, era, group, month):
        data.insert(0, 'era', era)
        data.insert(1, 'area_group', group)
        data.insert(2, 'month', month)

        return data

    def prepare_base_data(self, item):
        csv_data = item['csv_data']
        era = item['era']
        area_group = item['area_group']
        month = item['month']
        data = self.read_byte_csv(csv_data)
        base_data = self.add_info_to_data(data, era, area_group, month)
        del data
        return base_data
class DatabaseInsertProcessedData:
    def __init__(self, postgres_uri, base_table_name):
        self.postgres_uri = postgres_uri
        self.base_table_name = base_table_name

    @classmethod
    def from_crawler(cls, crawler):
        postgres_uri = crawler.settings.get("POSTGRES_URI")
        base_table_name = crawler.settings.get("POSTGRES_BASE_TABLE_NAME")
        if not postgres_uri:
            raise NotConfigured("DBのURIが正しく設定されていません")
        if not base_table_name:
            raise NotConfigured("DBへ登録されているデータファイル名参照テーブル名が正しく設定されていません")
        return cls(postgres_uri, base_table_name)

    def open_spider(self, spider):
        self.engine = create_engine(self.postgres_uri)

    def close_spider(self, spider):
        self.engine.dispose()

    def chose_insert_table(self, item):
        tmp_csv_path = item['csv_data']
        data = pd.read_csv(tmp_csv_path)
        era = data['era'][0]
        era_name = data['era_name'][0]
        era_initial = ''

        if era_name == '令和':
            era_initial = 'r'
        elif era_name == '平成':
            era_initial = 'h'
        elif era_initial == '昭和':
            era_initial = 's'
        else:
            raise DropItem('対応する元号がありません')

        table_name = f'{self.base_table_name}_{era_initial}_{era}'
        del data
        return table_name

    def process_item(self, item, spider):
        table_name = self.chose_insert_table(item)
        csv_data = pd.read_csv(item['csv_data'])
        try:
            csv_data.to_sql(table_name, self.engine, if_exists='append', index=False)
        except Exception as e:
            filename = item['filename']
            spider.logger.error(f'データ追加に失敗しました: {filename}, {e}')
            raise DropItem('DBへデータ追加失敗: 終了')

        del csv_data
        item['csv_data'].unlink()
        return item
