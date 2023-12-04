# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html

import io

import pandas as pd

from get_menu.preparation.preprocessor import data_processor

# useful for handling different item types with a single interface
# from itemadapter import ItemAdapter
# import re


class DataProcess:

    def process_item(self, item, spider):

        base_data = self.prepare_base_data(item)
        preprocessed_data = data_processor(base_data)

        item['csv_data'] = preprocessed_data.to_csv(index=False)
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
