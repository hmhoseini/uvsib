import uuid
from sqlalchemy import create_engine, Column, String, Text, Integer, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB, DOUBLE_PRECISION, ARRAY
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func
from uvsib.db.db_url import DB_URL

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
    model = Column(String, nullable=True)
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
    composition = Column(String, nullable=True)
    status = Column(String, nullable=False, default="Created")
    step_status = Column(JSONB, nullable=False, default={})
    stable_struct = Column(JSONB, nullable=True)
    attributes = Column(JSONB, nullable=True)
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
    composition = Column(String, nullable=True)
    chemsys = Column(String, nullable=True)
    attributes = Column(JSONB, nullable=True)
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
    method = Column(String, nullable=False)
    source = Column(String, nullable=True)
    structure = Column(JSONB, nullable=False)
    energy = Column(DOUBLE_PRECISION, nullable=True)
    vasprun_str = Column(Text, nullable=True)
    band_info = Column(JSONB, nullable=True)
    attributes = Column(JSONB, nullable=True)
    ctime = Column(DateTime(timezone=True), server_default=func.now())
    mtime = Column(DateTime(timezone=True), onupdate=func.now())

class DBSurface(Base):
    __tablename__ = "db_surface"

    id = Column(Integer, primary_key=True)
    structure_uuid = Column(
        UUID(as_uuid=True),
        ForeignKey("db_structure.uuid", ondelete="CASCADE"),
        nullable=False
    )
    slab = Column(JSONB, nullable=False) # structure & energy
    attributes = Column(JSONB, nullable=True)
    ctime = Column(DateTime(timezone=True), server_default=func.now())
    mtime = Column(DateTime(timezone=True), onupdate=func.now())

class DBSurfaceAdsorbate(Base):
    __tablename__ = "db_surface_adsorbate"

    id = Column(Integer, primary_key=True)
    surface_id = Column(
        Integer,
        ForeignKey("db_surface.id", ondelete="CASCADE"),
        nullable=False
    )
    reaction = Column(String, nullable=False)
    adsorbate = Column(String, nullable=False)
    structure = Column(JSONB, nullable=False)  # slab + adsorbate
    energy = Column(DOUBLE_PRECISION, nullable=False)  # adsorption energy
    attributes = Column(JSONB, nullable=True)
    ctime = Column(DateTime(timezone=True), server_default=func.now())
    mtime = Column(DateTime(timezone=True), onupdate=func.now())

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
    username = Column(String, nullable=True)
    composition = Column(String, nullable=True)
    model = Column(String, nullable=True)
    reaction = Column(String, nullable=True)
    status = Column(String, nullable=False, default="Created")
    step_status = Column(JSONB, nullable=True)
    attributes = Column(JSONB, nullable=True)
    ctime = Column(DateTime(timezone=True), server_default=func.now())
    mtime = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<db_test(uuid={self.uuid}, label={self.label})>"

if __name__ == "__main__":
    engine = create_engine(DB_URL, echo=False)
    Base.metadata.create_all(engine)
