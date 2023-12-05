from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.schema import Column
from sqlalchemy.types import String

Base = declarative_base()


class FilenameTable(Base):
    __tablename__ = 'filenames'
    filename = Column(String, primary_key=True)

