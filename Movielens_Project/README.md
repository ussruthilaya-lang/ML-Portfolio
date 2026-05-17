# Assignment1 - MovieLens Dataset Exploration
### CS6140 Machine Learning | Northeastern University
**Name:** Sruthilaya

---

## What is this?

This is my first assignment for CS6140. The goal was to explore the MovieLens 1M dataset and do some data analysis on it. Its basically looking at how users rate movies and what patterns show up when you break it down by age, gender, genre etc. At the end of it you can see how this kind of data could be used to build a recommendation system.

---

## Dataset

The MovieLens 1M dataset has 3 files:
- `ratings.dat` — userID, movieID, rating (1-5), timestamp
- `movies.dat` — movieID, title, genres
- `users.dat` — userID, gender, age, occupation, zip code

One thing to note - the age column is not actual age, its age group codes (1 = under 18, 25 = 25-34 etc). Took me a second to figure that out.

---

## Tasks Completed

**Q1 - Mean ratings by men aged above 25 per genre**
Filtered male users with age code > 25, merged with ratings and movies, exploded genres and grouped by genre to get mean ratings. Film-Noir came out on top which was surprising.

**Q2 - Top 5 movies by number of ratings**
Pretty straightforward groupby and count. American Beauty (1999) was #1 which makes sense since the dataset was collected around that time. Star Wars shows up 3 times in top 5 which is interesting from a franchise perspective.

**Q3 - Average ratings by age group**
Had to map the age codes to the groups the task asked for. Older users consistently rate higher than younger users - 51-70 group gave the highest average ratings.

**Q4 - Movies from same year as chosen movie, breakdown by reviewer age**
I picked Toy Story 2 (1999). Looked at all 1999 movies and broke down unique movies rated by 3 age buckets. Extended this to also compare how Toy Story 2 specifically performed vs the 1999 average - it was rated above average by every single age group which was cool to see.

**Q5 - Function to find similarly rated movies**
Wrote a function that takes user_id and movie_id, finds what rating that user gave the movie, then returns all other movies they rated the same way. Tested it with User 1 and Toy Story 2 and also tested the edge case where a user hasnt rated the movie at all.

**Q6 - Own insights**
Did 3 things here:
- **Prolific vs casual raters** - users who rate 500+ movies are almost 0.4 stars harsher on average than casual raters. More you watch, more critical you get.
- **Hidden gems** - used data-driven thresholds (25th percentile to median rating count) to find underseen but highly rated movies. Top result was Kurosawa's Sanjuro (4.61 avg) which most people have probably never heard of.
- **Genre vs Age Group heatmap** - visualized average rating for every genre x age group combination. Film-Noir is loved by everyone, Horror is disliked by everyone, and older audiences appreciate Animation way more than younger ones (nostalgia probably).

---

## Tools Used

- Python (Pandas, Matplotlib)
- Jupyter Notebook
- No SQL this time, did all the querying in Python which was a good exercise

---

## How to Run

1. Download the MovieLens 1M dataset from [here](https://github.com/khanhnamle1994/movielens)
2. Place `ratings.dat`, `movies.dat`, `users.dat` in the same folder as the notebook
3. Run all cells top to bottom

---

## Some things I noticed along the way

- American Beauty being #1 most rated looks like recency bias at first but when you check the timestamps all top 5 movies were rated around the same time (Oct 2000). So its more about Oscar hype + no vote splitting across a franchise rather than pure recency.
- The age codes in the dataset dont map perfectly to the age groups the homework asked for (like there is no code for 71+) so had to make some reasonable assumptions there.
- Most rated != best rated. The hidden gems analysis shows this clearly - the movies the algorithm would never recommend are sometimes the best ones.
- This was overall a pretty fun exploration project, had a lot of freedom to follow interesting directions in the data which I liked. Good foundation for understanding what goes into building a recommendation system before we get to the actual ML part.

---

## Files Submitted
- `exploration.ipynb` - main notebook
- `submission.pdf` - PDF export of notebook with all outputs