# Text Analysis of the 2017 UN Speeches

For this project, I scrape the entire corpus of speeches delivered during the 2017 UN General Assembly, as well as the official speech summaries as posted on the UN website. Unsupervised topic modeling produces several relevant issues that. Observing every country's dominant topic, as well as the distribution of important topics across the globe, gives relevant insight on national priorities, as well as strategic political placements.

In <a href="https://github.com/nico-gj/2017_un_speeches/blob/master/scripts/speech_scraper.ipynb" target="_blank">a first script</a>, I scrape the <a href="https://gadebate.un.org/en/sessions-archive" target="_blank">UN General Assembly Debate website</a> and collect every head of state's intervention, as well as the official statement summary. Statement summaries are saved as text files. Full statements are downloaded as PDF's, but converted to text using using `pdftotxt` from the <a href="https://www.xpdfreader.com/pdftotext-man.html" target="_blank">Xpdf suite</a>. I also conduct initial cleaning on the text files.

In <a href="https://github.com/nico-gj/2017_un_speeches/blob/master/scripts/2017_analysis.ipynb" target="_blank">a second script</a>, I conduct text analysis (PCA on TF-IDF word frequencies and LDA modeling) on the interventions and abstracts.

I give more information on the project and present interesting results <a href="https://nico-gj.github.io/post/2018/09/12/un-speeches.html" target="_blank">on this blog post</a>.

Feel free to get in touch if you have questions or comments !
Nico

Nicolas Guetta Jeanrenaud ([nicolas.jeanrenaud@gmail.com](nicolas.jeanrenaud@gmail.com))
