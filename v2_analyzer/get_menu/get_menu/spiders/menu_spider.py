import re
from pathlib import Path
from typing import List

from scrapy import Spider
from scrapy.http import Request

from v2_analyzer.get_menu.get_menu.items import GetMenuItem


class MenuSpider(Spider):
    name: str = 'menu_spider'

    allowed_domains = ['city.yokosuka.kanagawa.jp']
    start_urls: List[str] = ['https://www.city.yokosuka.kanagawa.jp/8345/kyuushoku/kyuusyoku-menu-open.html']
    csv_encoding = 'cp932'

    def parse(self, response):
        csv_links = response.css('a[href$=".csv"]::attr(href)').extract()
        for csv_link in csv_links:
            yield Request(url=response.urljoin(csv_link), callback=self.parse_csv)

    def parse_csv(self, response):

        csv_data = response.body
        filename = response.url.split('/')[-1]
        era, area_group, month = self.extract_info_from_filename(filename)
        item = GetMenuItem()
        item['csv_data'] = csv_data
        item['filename'] = filename
        item['era'] = era
        item['area_group'] = area_group
        item['month'] = month

        yield item

    def extract_info_from_filename(self, filename):
        info: str = re.split(r'od|\.', filename)[-2]
        area_group = info[-1]
        month = int(info[2:-1])
        if month > 12:
            month = int(str(month)[1])
        if month < 4:
            era = int(info[:2]) - 1
        else:
            era = int(info[:2])
        return era, area_group, month
