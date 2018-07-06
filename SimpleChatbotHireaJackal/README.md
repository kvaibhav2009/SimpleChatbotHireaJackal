# SimpleFAQchatbot


Need Python 3.6

It will automatically download the corpus.


Test it on postman client
Endpoint :
http://127.0.0.1:5000/getAnswer

{"query":"How much work experience do I need?"}

response:
{
"request-type": "ingest_data",
"response": {
"query": "reviews_file_path",
"answer 1": [
"Work experience includes opportunities in which students have been able to develop their professional and leadership skills. ",
"-200"
],
"answer 2": [
"Rather than focus on specific categories of work experiences, applicants should focus on their roles, responsibilities, and what they have learned from the types of work experiences that they have been involved in. The Admissions Board will look at the nature of the applicant's work experience when evaluating the applicants' ability to handle the academic rigor of our MBA program.",
"-300"
],
"answer 3": [
"The purpose of the interview is to better understand you as an MBA candidate for our program.  We encourage candidates to relax and be ready to talk about themselves.",
"-400"
]
}
}
