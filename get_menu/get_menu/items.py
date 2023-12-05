# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class GetMenuItem(scrapy.Item):
    # define the fields for your item here like:
    csv_data = scrapy.Field()
    filename = scrapy.Field()
    era = scrapy.Field()
    area_group = scrapy.Field()
    month = scrapy.Field()
