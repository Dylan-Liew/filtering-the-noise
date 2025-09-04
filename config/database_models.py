"""
SQLAlchemy models for the database schema.
Maps to CockroachDB tables used in the filtering pipeline.
"""

from sqlalchemy import BigInteger, String, Boolean, Column
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Data(Base):
    """Model for the preprocessed review data table."""
    __tablename__ = 'data'
    __table_args__ = {'schema': 'public'}
    
    # Primary key - auto-generated rowid
    rowid = Column(BigInteger, primary_key=True, autoincrement=True)
    
    # Review data columns
    review_time = Column(BigInteger, nullable=True)
    rating = Column(BigInteger, nullable=True) 
    uuid = Column(String, nullable=True)
    has_image = Column(Boolean, nullable=True)
    images = Column(String, nullable=True)
    review_cleaned = Column(String, nullable=True)
    business_description = Column(String, nullable=True)
    category = Column(String, nullable=True)
    
    def __repr__(self):
        return f"<Data(rowid={self.rowid}, uuid='{self.uuid}', rating={self.rating})>"