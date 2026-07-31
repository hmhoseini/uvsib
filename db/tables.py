import uuid
from sqlalchemy import create_engine, Column, String, Text, Integer, DateTime, ForeignKey, UniqueConstraint, Float, Boolean
from sqlalchemy.dialects.postgresql import UUID, JSONB, DOUBLE_PRECISION
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
    ehull = Column(DOUBLE_PRECISION, nullable=True)
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
    composition = Column(String, nullable=True)
    slab = Column(JSONB, nullable=False) # structure & energy
    attributes = Column(JSONB, nullable=True)
    ctime = Column(DateTime(timezone=True), server_default=func.now())
    mtime = Column(DateTime(timezone=True), onupdate=func.now())


class DBSurfaceAdsorbate(Base):
    __tablename__ = "db_surface_adsorbate"

    id = Column(Integer, primary_key=True)
    structure_uuid = Column(
        UUID(as_uuid=True),
        ForeignKey("db_structure.uuid", ondelete="CASCADE"),
        nullable=False
    )
    surface_id = Column(
        Integer,
        ForeignKey("db_surface.id", ondelete="CASCADE"),
        nullable=False
    )
    composition = Column(String, nullable=True)
    reaction = Column(String, nullable=False)
    reaction_path = Column(String, nullable=False)
    site_type = Column(String, nullable=False)
    ads_coord = Column(Text, nullable=False)
    repeat = Column(Text, nullable=False)
    eta = Column(DOUBLE_PRECISION, nullable=True)
    dG = Column(JSONB, nullable=True)
    adsorb_set = Column(JSONB, nullable=False) # structures & energies
    attributes = Column(JSONB, nullable=True)
    ctime = Column(DateTime(timezone=True), server_default=func.now())
    mtime = Column(DateTime(timezone=True), onupdate=func.now())


class DBSurfaceMLAdsorbate(Base):
    __tablename__ = "db_surface_ml_adsorbate"

    id = Column(Integer, primary_key=True)
    structure_uuid = Column(UUID(as_uuid=True), ForeignKey("db_structure.uuid", ondelete="CASCADE"), nullable=False)
    surface_id = Column(Integer, ForeignKey("db_surface.id", ondelete="CASCADE"), nullable=False)
    surface_miller_index = Column(JSONB, nullable=False)
    composition = Column(String, nullable=True)
    reaction = Column(String, nullable=False)
    reaction_path = Column(String, nullable=False)
    site_type = Column(String, nullable=False)
    ads_coord = Column(JSONB, nullable=False)
    repeat = Column(Text, nullable=False)
    eta = Column(DOUBLE_PRECISION, nullable=False)
    dG_steps = Column(JSONB, nullable=False)
    dG_cumulative = Column(JSONB, nullable=False)
    adsorb_set = Column(JSONB, nullable=False) # structures & energies
    attributes = Column(JSONB, nullable=True)
    ctime = Column(DateTime(timezone=True), server_default=func.now())
    mtime = Column(DateTime(timezone=True), onupdate=func.now())


class DBBatteryPath(Base):
    """Battery (deintercalation) characteristics, one row per
    (composition, working_ion, host structure). Written by BatteryWorkChain;
    ``voltage_profile``/``configs`` hold the hull + the relaxed configuration
    set (structures & energies) so tier-2 stages (DFT verification, NEB) can
    reuse them without re-enumerating."""
    __tablename__ = "db_battery_path"

    id = Column(Integer, primary_key=True)
    structure_uuid = Column(
        UUID(as_uuid=True),
        ForeignKey("db_structure.uuid", ondelete="CASCADE"),
        nullable=False
    )
    composition = Column(String, nullable=True)
    working_ion = Column(String, nullable=False)
    model = Column(String, nullable=True)
    avg_voltage = Column(DOUBLE_PRECISION, nullable=False)
    capacity_grav = Column(DOUBLE_PRECISION, nullable=False)   # mAh/g
    capacity_vol = Column(DOUBLE_PRECISION, nullable=False)    # mAh/cm^3
    energy_density = Column(DOUBLE_PRECISION, nullable=False)  # Wh/kg
    volume_change_pct = Column(DOUBLE_PRECISION, nullable=False)
    endpoint_ehull = Column(DOUBLE_PRECISION, nullable=True)   # charged host
    voltage_profile = Column(JSONB, nullable=False)  # vertices/steps/points
    configs = Column(JSONB, nullable=False)          # relaxed structures & energies
    flags = Column(JSONB, nullable=True)             # framework_changed, ...
    attributes = Column(JSONB, nullable=True)
    ctime = Column(DateTime(timezone=True), server_default=func.now())
    mtime = Column(DateTime(timezone=True), onupdate=func.now())


class DBBatteryNEB(Base):
    """Ion-migration barriers for one (composition, working_ion, host,
    migration limit). Written by BatteryNEBWorkChain; ``hops`` holds the
    symmetry-distinct barriers (+ TS structures), the e_m_* columns the
    percolation thresholds (the lowest barrier at which the hop network
    wraps the cell in 1/2/3 independent directions)."""
    __tablename__ = "db_battery_neb"

    id = Column(Integer, primary_key=True)
    structure_uuid = Column(
        UUID(as_uuid=True),
        ForeignKey("db_structure.uuid", ondelete="CASCADE"),
        nullable=False
    )
    composition = Column(String, nullable=True)
    working_ion = Column(String, nullable=False)
    hop_limit = Column(String, nullable=False)     # 'vacancy' | 'dilute'
    model = Column(String, nullable=True)
    e_m_1d = Column(DOUBLE_PRECISION, nullable=True)
    e_m_2d = Column(DOUBLE_PRECISION, nullable=True)
    e_m_3d = Column(DOUBLE_PRECISION, nullable=True)
    hops = Column(JSONB, nullable=False)           # per-class barriers, TS
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
    reaction = Column(String, nullable=True)
    reaction_path = Column(String, nullable=False)
    nano_particles = Column(String, nullable=True)
    status = Column(String, nullable=False, default="Created")
    step_status = Column(JSONB, nullable=True)
    result = Column(String, nullable=True)
    attributes = Column(JSONB, nullable=True)
    ctime = Column(DateTime(timezone=True), server_default=func.now())
    mtime = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<db_test(uuid={self.uuid}, label={self.label})>"


class DBNanoParticles(Base):
    """Frontend table"""
    __tablename__ = 'db_nano_particles'
    uuid = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        unique=True,
        nullable=False
    )
    num_atoms = Column(Integer, nullable=True)
    elements = Column(String, nullable=True)
    energy = Column(DOUBLE_PRECISION, nullable=True)
    special_type = Column(String, nullable=True)
    structure = Column(JSONB, nullable=True)
    model = Column(String, nullable=True)
    status = Column(String, nullable=False, default="Created")
    step_status = Column(JSONB, nullable=False, default=dict({}))
    attributes = Column(JSONB, nullable=True)
    ctime = Column(DateTime(timezone=True), server_default=func.now())
    mtime = Column(DateTime(timezone=True), onupdate=func.now())


class DBFinetuneFrame(Base):
    """DFT-label candidate frames for the solvated catalysis NEB fine-tune
    (active learning). Written by db/ingest_frames.py from the output of
    codes/files/solvation_frames.py; exported in a single batch by
    run_dir/export_all.py --finetune-frames. `bulk_uuid`/`surface_id` carry
    the attribution back to the frozen substrate set (no FK on purpose --
    frames may be ingested into a different DB than the run that produced
    the substrate). `status` tracks the AL bookkeeping: new -> exported
    (handed to DFT) -> labeled (energies/forces computed)."""
    __tablename__ = "db_finetune_frames"

    id = Column(Integer, primary_key=True)
    batch = Column(String, nullable=False, index=True)   # e.g. 'cuau-gen0'
    generation = Column(Integer, nullable=False, default=0)
    kind = Column(String, nullable=False)   # md_snapshot|neb_image|neb_endpoint
    composition = Column(String, nullable=True)
    model = Column(String, nullable=True)                # harvesting MLIP
    surface_id = Column(Integer, nullable=True)
    bulk_uuid = Column(UUID(as_uuid=True), nullable=True)
    reaction = Column(String, nullable=True)
    reaction_path = Column(String, nullable=True)
    structure = Column(JSONB, nullable=False)            # pmg Structure dict
    energy_model = Column(DOUBLE_PRECISION, nullable=True)
    status = Column(String, nullable=False, default="new")
    attributes = Column(JSONB, nullable=True)            # meta (barriers, ...)
    ctime = Column(DateTime(timezone=True), server_default=func.now())
    mtime = Column(DateTime(timezone=True), onupdate=func.now())


class DBSimilarities(Base):
    __tablename__ = "db_similarities"
    uuid = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, unique=True, nullable=False)
    similarity = Column(Float, nullable=True)
    composition = Column(String, nullable=True)
    csp_structure = Column(JSONB, nullable=True)
    chemical_system = Column(String, nullable=True)
    reference_structure = Column(JSONB, nullable=True)
    reference_material_id = Column(String, nullable=True)
    mtime = Column(DateTime(timezone=True), onupdate=func.now())
    ctime = Column(DateTime(timezone=True), server_default=func.now())


class DBInterface(Base):
    """One electrode|electrolyte half-cell junction.

    Written by InterfaceWorkChain. Rows exist for BOTH outcomes of the stage-1
    thermodynamic screen: a pair that reacts is recorded with
    ``built = False`` and no structure, because "this pair decomposes" is a
    result worth keeping, not an absence. See docs/interfaces.md.

    ``active_mask`` marks the atoms within the junction region; downstream
    relaxation and NEB freeze the rest.
    """
    __tablename__ = "db_interface"

    id = Column(Integer, primary_key=True)
    # nullable: a pair rejected by the stage-1 screen never gets a structure
    structure_uuid = Column(
        UUID(as_uuid=True),
        ForeignKey("db_structure.uuid", ondelete="CASCADE"),
        nullable=True
    )
    composition = Column(String, nullable=True)        # parent electrode formula
    electrode = Column(String, nullable=False)
    electrolyte = Column(String, nullable=False)
    working_ion = Column(String, nullable=False)
    half_cell = Column(String, nullable=False)         # 'anode' | 'cathode'
    model = Column(String, nullable=True)

    # --- stage 1: pseudo-binary interface reaction (Chem. Mater. 28, 266) ---
    reaction_energy = Column(DOUBLE_PRECISION, nullable=True)  # eV/atom, <= 0
    reaction_products = Column(JSONB, nullable=True)
    reacts = Column(Boolean, nullable=True)
    severe = Column(Boolean, nullable=True)
    mu_worst = Column(DOUBLE_PRECISION, nullable=True)  # mu_ion of the worst case
    reaction_scan = Column(JSONB, nullable=True)        # the whole mu sweep

    # --- stage 2: Zur-McGill geometry (null when the pair was rejected) -----
    built = Column(Boolean, nullable=False, default=False)
    # The UNRELAXED junction geometry lives here rather than in db_structure:
    # a db_structure row implies a computed energy, and these have none until
    # the relaxation stage runs (which then fills structure_uuid).
    structure = Column(JSONB, nullable=True)
    film_miller = Column(JSONB, nullable=True)
    substrate_miller = Column(JSONB, nullable=True)
    termination = Column(JSONB, nullable=True)
    n_atoms = Column(Integer, nullable=True)
    area = Column(DOUBLE_PRECISION, nullable=True)
    strain_percent = Column(DOUBLE_PRECISION, nullable=True)
    active_mask = Column(JSONB, nullable=True)

    attributes = Column(JSONB, nullable=True)
    ctime = Column(DateTime(timezone=True), server_default=func.now())
    mtime = Column(DateTime(timezone=True), onupdate=func.now())


if __name__ == "__main__":
    engine = create_engine(DB_URL, echo=False)
    Base.metadata.create_all(engine)
