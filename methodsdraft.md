II. Method
	a. Using data from a previous study which did similar effects
		i. The previous study did ....
		ii. (Reivew data collection and study design)
	b. Correlation matrix involving relevant autonomic and subjective features.
	c. We used multi level models implemented by the MixedModels package in the Julia language.
	d. For each sets of trials conditioned on a specific stimulus, we created models to predict a subject's corresponding trait phobia from within-subject factors, between-subject factors, and interaction factors involving phobias. The features within these factors each capture autonomic arousal, skin conductance and heart rate, and subjective aurosal and valence. 

We first conducted a correlation analysis in which we dropped missing rows and analyzed relevant autonomic and subjective features. This included the heights, social and spiders phobia measures, the subjective measures, arousal fear and valence and the autonomic features, scr, hp each separated and aggregated over the associated trial stimulus conditions.

Using the MixedModels package available to the Julia language, we then created hierarchical models to predict a subject's fear of heights ratings from within-subject factors, between-subject factors, and interaction factors involving phobias for each stimulus category. These factors include autonomic measures, skin conductance and heart rate, and subjective arousal and valence. We then repeated this analysis for both their fear of spiders ratings and their social fear ratings.


