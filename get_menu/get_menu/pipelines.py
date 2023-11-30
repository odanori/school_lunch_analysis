# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html

import io
# useful for handling different item types with a single interface
# from itemadapter import ItemAdapter
import re

import pandas as pd

from get_menu.preparation.preprocessor import data_processor


class DataProcessPipeline:

    def process_item(self, item, spider):
        csv_data = item['csv_data']
        filename = item['filename']

        base_data = self.prepare_base_data(csv_data, filename)
        preprocessed_data = data_processor(base_data)

        item['csv_data'] = preprocessed_data.to_csv(index=False)
        return item

    def read_byte_csv(self, csv_data):
        data = pd.read_csv(io.BytesIO(csv_data), encoding='cp932', index_col=None)
        return data

    def extract_info_from_filename(self, filename):
        info: str = re.split(r'od|\.', filename)[-2]
        group = info[-1]
        month = int(info[2:-1])
        if month > 12:
            month = int(str(month)[1])
        if month < 4:
            era = int(info[:2]) - 1
        else:
            era = int(info[:2])
        return era, group, month

    def add_info_to_data(self, data, era, group, month):
        data.insert(0, 'era', era)
        data.insert(1, 'group', group)
        data.insert(2, 'month', month)
        return data

    def prepare_base_data(self, csv_data, filename):
        data = self.read_byte_csv(csv_data)
        era, group, month = self.extract_info_from_filename(filename)
        base_data = self.add_info_to_data(data, era, group, month)
        del data
        return base_data
