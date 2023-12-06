

class Config:
    # DB接続情報
    POSTGRES_URI = 'postgresql://{user}:{pass}@{host}:{port}/lunch_menu'
    POSTGRES_FILENAME_TABLE = 'filenames'
    POSTGRES_BASE_TABLE_NAME = 'menu'
