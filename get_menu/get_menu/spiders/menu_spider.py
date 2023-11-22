from pathlib import Path
from typing import List

from preparation.reader import read_data
from scrapy import Spider
from scrapy.http import Request


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
            headers = {'Accept-Encoding': 'identity'}
            yield Request(url=response.urljoin(csv_link), callback=self.save_csv, headers=headers)

    def save_csv(self, response):
        filename = response.url.split('/')[-1]
        save_path = self.download_path / filename
        with open(save_path, 'wb') as f:
            f.write(response.body)
        self.log(f'saved file {save_path}')

        data = read_data(save_path)
        print(data)

