import uuid
from sqlalchemy import create_engine, Column, String, Integer, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB, DOUBLE_PRECISION
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func
from db.db_url import DB_URL

Base = declarative_base()

class DBChemsys(Base):
    """Chemical system table"""
    __tablename__ = "db_chemsys"

    uuid = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        unique=True,
        nullable=False
    )
    chemsys = Column(String, nullable=False)
    gen_structures = Column(String, nullable=True)

    __table_args__ = (
        UniqueConstraint("chemsys", name="_list_formula_uc"),
    )

class DBComposition(Base):
    """Composition table """
    __tablename__ = "db_composition"

    uuid = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        unique=True,
        nullable=False
    )
    label = Column(String, nullable=True, default="composition")
    description = Column(String, nullable=True)
    attributes = Column(JSONB, nullable=True)
    composition = Column(String, nullable=True)
    struct_ehull = Column(JSONB, nullable=True)
    stable_struc = Column(JSONB, nullable=True)
    status = Column(String, nullable=False, default="Created")
    step_status = Column(JSONB, nullable=True)
    extras = Column(JSONB, nullable=True)
    parent_uuid = Column(UUID(as_uuid=True), nullable=True)
    child_uuid = Column(UUID(as_uuid=True), nullable=True)
    ctime = Column(DateTime(timezone=True), server_default=func.now())
    mtime = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<db_test(uuid={self.uuid}, label={self.label})>"

class DBStructure(Base):
    __tablename__ = "db_structure"
    uuid = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        unique=True,
        nullable=False
    )
    label = Column(String, nullable=True, default="structure")
    description = Column(String, nullable=True)
    composition = Column(String, nullable=True)
    chemsys = Column(String, nullable=True)
    attributes = Column(JSONB, nullable=True)
    status = Column(String, nullable=False, default="Created")
    extras = Column(JSONB, nullable=True)
    parent_uuid = Column(UUID(as_uuid=True), nullable=True)
    child_uuid = Column(UUID(as_uuid=True), nullable=True)
    ctime = Column(DateTime(timezone=True), server_default=func.now())
    mtime = Column(DateTime(timezone=True), onupdate=func.now())

class DBStructureVersion(Base):
    __tablename__ = 'db_structure_version'

    id = Column(Integer, primary_key=True)
    structure_uuid = Column(
        UUID(as_uuid=True),
        ForeignKey("db_structure.uuid", ondelete="CASCADE"),
        nullable=False
    )
    method = Column(String, nullable=False)  # e.g., "ML", "PBE", "SCAN", etc.
    structure = Column(JSONB, nullable=False)
    energy = Column(DOUBLE_PRECISION, nullable=True)
    epa = Column(DOUBLE_PRECISION, nullable=True)
    attributes = Column(JSONB, nullable=True)
    ctime = Column(DateTime(timezone=True), server_default=func.now())

class DBFrontend(Base):
    """Frontend table"""
    __tablename__ = 'db_frontend'
    uuid = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        unique=True,
        nullable=False
    )
    label = Column(String, nullable=True, default="frontend")
    description = Column(String, nullable=True)
    attributes = Column(JSONB, nullable=True)
    username = Column(String, nullable=True)
    composition = Column(String, nullable=True)
    status = Column(String, nullable=False, default="Created")
    step_status = Column(JSONB, nullable=True)
    extras = Column(JSONB, nullable=True)
    parent_uuid = Column(UUID(as_uuid=True), nullable=True)
    child_uuid = Column(UUID(as_uuid=True), nullable=True)
    exit_status = Column(Integer, nullable=True)
    ctime = Column(DateTime(timezone=True), server_default=func.now())
    mtime = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<db_test(uuid={self.uuid}, label={self.label})>"

if __name__ == "__main__":
    engine = create_engine(DB_URL, echo=False)
    Base.metadata.create_all(engine)
