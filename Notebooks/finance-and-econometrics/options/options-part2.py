import marimo

__generated_with = "0.25.2"
app = marimo.App(css_file="../../_theme/blackboard.css")

with app.setup:
    # Import libraries and set up the notebook in the initialisation

    # Imports for notebook set-up, plots and themes:
    import marimo as mo
    import matplotlib.pyplot as plt

    from mathsandmeander.common import bib, theme, xref

    theme.apply(usetex=True)

    # Imports for numerical and data analysis:
    import numpy as np
    import pandas as pd


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Financial Theory – Options (Part II):
    By Donnell Obovu

    ### I. Preface:

    This is Part II of a four-part series of interactive notebooks whose purpose is to introduce financial options.
    Each part works through a different aspect of option theory:
    * Part I: Introduction to Options
    * Part II: Working with Data on Options
    * Part III: Methods for Pricing Options
    * Part IV: Strategies for Trading Options

    ### II. Prerequisites:
    For this notebook, I recommend that readers have an understanding of:
    - Part I of this series

    ### III. Contents:
    * [3. Working with Option Market Data](#sec-market-data)
        * [3.1. Anatomy of an Option Chain – Bid, Ask, Mid, Volume, Open Interest](#sec-option-chain)
        * [3.2. Data Cleaning and Sanity Checks](#sec-data-cleaning)
        * [3.3. Testing Put-Call Parity on Real Quotes](#sec-parity-test)
        * [3.4. Backing Out the Implied Forward and Dividend Yield](#sec-implied-forward)

    ### IV. Notation:
    The notation follows the notation table in Part I.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    <span id="sec-market-data"></span>
    ## 3. Working with Option Market Data:
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    <span id="sec-option-chain"></span>
    ### 3.1. Anatomy of an Option Chain – Bid, Ask, Mid, Volume, Open Interest:
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    <span id="sec-data-cleaning"></span>
    ### 3.2. Data Cleaning and Sanity Checks:
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    <span id="sec-parity-test"></span>
    ### 3.3. Testing Put-Call Parity on Real Quotes:
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    <span id="sec-implied-forward"></span>
    ### 3.4. Backing Out the Implied Forward and Dividend Yield:
    """)
    return


@app.cell(hide_code=True)
def _():
    bib.bibliography(__file__)
    return


if __name__ == "__main__":
    app.run()
