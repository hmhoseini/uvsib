# UvSiB — Conversion of Solar Energy into Fuels

**UvSiB** is an open, workflow-driven platform for accelerating the discovery and evaluation of photocatalytic materials for solar-fuel production.

The project combines high-throughput atomistic simulation, machine-learning models, materials databases, and automated scientific workflows. Its long-term goal is to give researchers in materials science, chemistry, and physics a user-friendly way to design virtual experiments, screen candidate materials, and identify promising photocatalysts without requiring specialist expertise in artificial intelligence or workflow automation.

> **Project status:** Active research software under development. APIs, workflows, database schemas, and installation procedures may change.

## Why UvSiB?

Solar energy can, in principle, drive the production of fuels and fundamental chemicals from abundant feedstocks such as water and carbon dioxide. Practical photocatalysts, however, must satisfy several requirements simultaneously:

- suitable light absorption and electronic structure;
- efficient charge separation and transport;
- favorable surface reaction energetics;
- chemical and structural stability;
- low cost, scalability, and synthesizability.

Experimental trial-and-error across the enormous chemical and structural search space is expensive and slow. UvSiB addresses this challenge by connecting materials generation, machine-learning screening, first-principles verification, surface modelling, and reaction analysis in reproducible computational workflows.

## Project vision

UvSiB is being developed as an open-access virtual materials-design environment with three central capabilities:

1. **High-throughput simulation** — automated calculations over large sets of candidate materials.
2. **Machine-learning-assisted discovery and analysis** — faster structure generation, relaxation, screening, and mapping between structures and properties.
3. **Composable scientific workflows** — reusable computational building blocks that can be combined into virtual experiments and executed with minimal manual intervention.

The platform is intended to support the discovery of materials for solar-fuel reactions, including water splitting and carbon-dioxide conversion, and to provide reusable data, models, and workflows to the wider community.

## Repository scope

This repository contains the computational backend and workflow components of UvSiB. The current codebase includes:

### Materials generation and exploration

- crystal-structure prediction and generation workflows;
- integrations for **MatterGen** and **GNoME**;
- minima-hopping and similarity-analysis tools;

### Machine-learning interatomic potentials

AiiDA calculation, parser, and workchain integrations are provided for several machine-learning models, including:

- MatterSim;
- MACE;
- uPET;
- UMA.

These models can be used for rapid structure relaxation and screening before more expensive first-principles verification.

### Surfaces and photocatalytic reactions

- surface and slab construction;
- adsorption-site and adsorbate generation;
- reaction-path and free-energy data models;
- workflows for catalytic-reaction analysis.

## Software architecture

UvSiB is structured as an **AiiDA plugin**. The main components are:

```text
uvsib/
├── codes/          # AiiDA calculations, parsers, workchains, and executable templates
├── workchains/     # Higher-level scientific workflows
├── workflows/      # Workflow orchestration and settings
├── db/             # SQLAlchemy database models and utility functions
├── docs/           # Developer and installation notes
├── setup.py
└── setup.json      # Package metadata, dependencies, and AiiDA entry points
```

The registered AiiDA entry points cover structure generation, ML potentials, minima hopping, similarity analysis, SQS, phonons, phase diagrams, band alignment, surface construction, adsorbates, and related workflows.

## Requirements

The package currently declares:

- Python 3.10 or newer;
- AiiDA and `aiida-pythonjob`;
- ASE;
- pymatgen;
- Materials Project API client (`mp-api`);
- `aiida-submission-controller`.

Individual workflows may require additional external codes, trained model files, pseudopotentials, databases, scheduler configurations, and AiiDA computer/code registrations.

## Installation

Clone the repository and install it in an isolated Python environment:

```bash
git clone https://github.com/hmhoseini/uvsib.git
cd uvsib

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .
```

Confirm that AiiDA can discover the plugin entry points:

```bash
verdi plugin list aiida.calculations
verdi plugin list aiida.workflows
verdi plugin list aiida.parsers
```

Before running workflows, configure:

1. an AiiDA profile;
2. one or more local or remote computers;
3. the required simulation codes and ML executables;
4. PostgreSQL/database access where needed;
5. API keys and project-specific settings.

Because the repository is under active development, consult the source code and files in `docs/` for workflow-specific configuration.

## Reproducibility and provenance

AiiDA records calculation inputs, outputs, codes, computational resources, and workflow dependencies as a provenance graph. UvSiB builds on this capability so that multi-stage materials-discovery campaigns can be inspected, reproduced, restarted, and extended.

The separate UvSiB database stores domain-level entities and curated results needed by the platform, while AiiDA retains detailed computational provenance.

## Roadmap

Planned development directions include:

- a user-friendly open web interface;
- reusable end-to-end photocatalyst-screening workflows;
- expanded initial datasets and pretrained models;
- improved workflow composition and monitoring;
- broader support for water splitting and carbon-dioxide conversion;
- stronger links between calculated candidates and experimental validation;
- documentation, examples, tests, and reproducible demonstration cases.

## Funding and project period

UvSiB is conducted at the **Center for Advanced Systems Understanding (CASUS)**, an institute of the **Helmholtz-Zentrum Dresden-Rossendorf (HZDR)**.

The project is supported through the European Union's **Just Transition Fund (JTF)** and with participation from the Free State of Saxony under the *Forschung InfraProNet 2021–2027* funding framework.

## License

The package metadata identifies this repository as released under the **MIT License**. Add a root-level `LICENSE` file to make the licensing terms explicit for users and contributors.

## Acknowledgment

When presenting or reusing UvSiB, please acknowledge CASUS at HZDR, the European Union's Just Transition Fund, and the Free State of Saxony.
