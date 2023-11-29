from pathlib import Path
from typing import List

from scrapy import Spider
from scrapy.http import Request

from get_menu.get_menu.items import GetMenuItem


class MenuSpider(Spider):
    name: str = 'menu_spider'

    allowed_domains = ['city.yokosuka.kanagawa.jp']
    start_urls: List[str] = ['https://www.city.yokosuka.kanagawa.jp/8345/kyuushoku/kyuusyoku-menu-open.html']
    csv_encoding = 'cp932'

    download_path = Path('./tmp')
    if not download_path.exists():
        download_path.mkdir()

    def parse(self, response):
        csv_links = response.css('a[href$=".csv"]::attr(href)').extract()
        for csv_link in csv_links:
            # headers = {'Acceept-Encoding': 'identity'}
            # yield Request(url=response.urljoin(csv_link), callback=self.parse_csv, headers=headers)
            yield Request(url=response.urljoin(csv_link), callback=self.parse_csv)

    def parse_csv(self, response):
        csv_data = response.body
        filename = response.url.split('/')[-1]
        # print(pd.read_csv(io.BytesIO(csv_data), encoding='cp932'))
        item = GetMenuItem()
        item['csv_data'] = csv_data
        item['filename'] = filename
        yield item


    # def parse(self, response):
    #     csv_links = response.css('a[href$=".csv"]::attr(href)').extract()

    #     for csv_link in csv_links:
    #         headers = {'Accept-Encoding': 'identity'}
    #         yield Request(url=response.urljoin(csv_link), callback=self.data_preparation, headers=headers)

    # def save_csv(self, response) -> Path:
    #     filename = response.url.split('/')[-1]
    #     save_path = self.download_path / filename
    #     with open(save_path, 'wb') as f:
    #         f.write(response.body)
    #     self.log(f'saved file {save_path}')
    #     return save_path

    # def process_csv(self, data_path: Path) -> InfoAndData:
    #     data_info = read_data(data_path)
    #     preprocessed_data = data_processor(data_info.data)
    #     preprocessed_data_info = InfoAndData(data_info.era, data_info.month, data_info.group, preprocessed_data)
    #     return preprocessed_data_info

    # def data_preparation(self, response):
    #     save_path = self.save_csv(response)
    #     preprocessed_data_info = self.process_csv(save_path)
    #     print(preprocessed_data_info)
