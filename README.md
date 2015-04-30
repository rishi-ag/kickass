For homework #1

Team: Rishabh Agnihotri, Samantha Dalton, Jordan McIver

Question 3 is answered in q3_regex.ipynb

Question 4: Scraping Kickstarter for successful project information
	   Scripts: 
		A) project_metadata.ipynb--scrapes the project search page for individual
					kickstarte campain urls and blurb information
					(about 20 projects on each page until need to 						click load more button)
				  Produces data files:
				       	1.) project_successful_meta.json-successful 						project URLS gathered only---only output data 						from this script used for hwk1
					2.)project_live_meta.json--live project URLS 						only--not used in hwk 1 (but will be used for 						final project)
					3.)project_meta.json---gathers URL of all projects (live and 						successful)--not used in hwk1 (but used later)
		B) page_scrape_successful_projects.ipynb --- scrapes individual project URL pages by looping 								  through succesful projects' URLS collected in the 							  	  project_metadata.ipynb program
				Produces data file:
					1.)project_successful_detail.json ---detailed info about successful 										      kickstarter projects
						****Note: Only included 500 project records from scraped 								URLS for homework  submission, but will have 								more/all for final project

Github repository: https://github.com/rishi1226/kickass.git

