from typing import Any, Text, Dict, List
from transformers import pipeline
from datetime import datetime

import pdfkit

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher


class ActionDefaultFallback(Action):
    def name(self) -> Text:
        return "action_default_fallback"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        dispatcher.utter_message("I'm sorry, I didn't understand that. Can you please rephrase?")
        return []

class ActionHelloWorld(Action):

    def name(self) -> Text:
        return "action_hello_world"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        dispatcher.utter_message(text="Hello World!")

        return []
    
class ActionGenerateReport(Action):

    def name(self) -> Text:
        return "action_generate_report"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

        #stores all the messages of the users
        user_messages = [event.get("text") for event in tracker.events if event["event"] == "user"]

        #running user messages into the classifier to get the results
        model_outputs = classifier(user_messages)
        report = model_outputs[0]

        #path to store the pdf
        pdf_filename = r"C:\Users\ADMIN\Desktop\emotion_report.pdf"
        dynamic_html = generate_dynamic_html(report)
        save_html_to_pdf(dynamic_html, pdf_filename)

        dispatcher.utter_message(text=f"Bye bye! If you ever need assistance, I'll be here. Also, here is a report on your emotional wellness:\n{pdf_filename}")

        return []
    
def generate_dynamic_html(report: List[Dict[Text, Any]]) -> str:
    # Define the dynamic HTML content
    # Extract labels and scores
    labels = [emotion['label'] for emotion in report]
    scores = [emotion['score'] for emotion in report]

    # Sort data by scores in descending order
    sorted_indices = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_scores = [scores[i] for i in sorted_indices]

    name='Kelvin'
    time=(datetime.now()).strftime("%H:%M:%S %Y-%m-%d")

    # Generate the HTML for the bar graph
    html_content = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: "Helvetica", sans-serif;
                font-size: 12px;
            }}
            h1 {{
                color: #51c054;
            }}
            .report-container {{
                padding: 2rem;
                margin: 2rem;
                width: 48rem;
                margin-left: auto;
                margin-right: auto;
            }}
            .bar-graph {{
                display: flex;
                flex-direction: column;
                align-items: start;
                margin-top: 1rem;
            }}
            .bar {{
                height: 1rem;
                margin-bottom: 0.5rem;
                flex-grow: 1;
                border-radius: 0.5rem;
                background-color: #f0f0f0;
                background-color: #51c054;
            }}
            .row {{
                display: flex;
                align-items: center;
                flex-direction: row;
                gap: 1rem;
            }}
            .label{{
                font-weight: 600;
            }}
        </style>
    </head>
    <body>
         <div class="report-container">
            <div>
                <h1>Emotional Detection Report</h1>
                <p>Name: {name}</p>
                <p>Time: {time}</p>
                <p>Dominant Emotion: {sorted_labels[0]}</p>
            </div>
             <div class="bar-graph">
                {"".join([f"<div style='width: 100%'><p>{int(score * 100)}% - {label}</p><div style='width: 100%''><div class='bar' style='width: {score * 100}%;'></div></div>" for i, (label, score) in enumerate(zip(sorted_labels, sorted_scores))])}
            </div>
        </div>
    </body>
    </html>
    """

    return html_content

def save_html_to_pdf(html_content: str, pdf_filename: str):
    path_wkhtmltopdf = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
    config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)
    pdfkit.from_string(html_content, pdf_filename, configuration=config)
    
        
    
    
    
