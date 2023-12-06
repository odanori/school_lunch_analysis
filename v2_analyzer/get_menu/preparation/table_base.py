from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.schema import Column
from sqlalchemy.types import REAL, Date, Integer, String, Text

Base = declarative_base()


class FilenameTable(Base):
    __tablename__ = 'filenames'
    filename = Column(String, primary_key=True)


menu_table_type = {
    'era': Integer,
    'area_group': Text,
    'month': Integer,
    'date': Date,
    'menu': Text,
    'ingredient': Text,
    'amount_g': REAL,
    'carolies_kcal': REAL,
    'protein_g': REAL,
    'fat_g': REAL,
    'sodium_mg': REAL,
    'era_name': Text
    }
