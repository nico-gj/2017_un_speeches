# Analysis of the 2017 UN Speeches

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/nico-gj/2017_un_speeches/master)

## Introduction and Main Results

This project applies text analysis methods to the United Nations' 72nd Session General Assembly Debates. Input data is a set of over 150 English versions of the speeches made by Heads of States. I calculate summary statistics, and run a Latent-Dirichlet Allocation (LDA) model to cluster the speeches by the dominant issues.

Among the 13 topics I generate, at least 4 refer to very clear geo-political issues (North Korea, the Arab Region, Climate Change, Human Rights in Russia). The other topics seem more broad and hardly interpretable – but I may be missing something as well ! Since the speeches
Once I assign a dominant topic to each speech, I can identify a given countries national priorities. The countries most concerned with North Korea (Japan, South Korea) and Climate Change (Dominica, Fiji, Micronesia, New Zealand, and Papua-New Guinea) are unsurprising. The two other topics seem a little broad to draw any conclusions.


## Technical Note

### UN Speeches Scraper

The first script scrapes the UN website and downloads speeches from the 72nd Session General Assembly Debates. I download only the English translation of the speech, and therefore restrict to countries where an English version is available.  All speeches are downloaded in `.pdf` format.

Conversion to from `.pdf` to `.txt` is done using the command line (`pdftotxt` from the `xpdf` suite).

### Data Cleaning

The second script converts all `.txt` files to a DataFrame, which I export in `.csv` format.

### Analysis

The third script performs text analysis on the DataFrame. Preliminary analyses include word counts, as well as comparison visualizations between 2 countries.

I also run an LDA Topic Model on the data. The most relevant model I come up with is a 13-component model, but other distributions may offer interesting insight as well. I then test how well these topics explain the data. This final part is interactive and I invite users to come up with new results !


## Contact Information

Nicolas Guetta Jeanrenaud ([nicolas.jeanrenaud@gmail.com](nicolas.jeanrenaud@gmail.com))
