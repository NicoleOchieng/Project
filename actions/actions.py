from typing import Any, Text, Dict, List
from datetime import datetime
from ktrain import load_predictor

import pdfkit
import time

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from keras.models import load_model


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
        
        predictor = load_predictor(r"C:\Users\ADMIN\Project")
        # text_to_predict = "This is an example text for emotion classification."
        text_to_predict = [event.get("text") for event in tracker.events if event["event"] == "user"]
        paragraph = ". ".join(text_to_predict)
        predicted_emotion= ''
        predicted_class = predictor.predict(paragraph)

        emotion_mapping = {
            0: "sadness",
            1: "joy",
            2: "love",
            3: "anger",
            4: "fear",
            5: "surprise"
        }

        if predicted_class in emotion_mapping:
            predicted_emotion = emotion_mapping[predicted_class]
        else:
            print("Invalid predicted class.")

    
        print(f"Predicted emotion: {predicted_emotion}")
        
        pdf_filename = r"C:\Users\ADMIN\Desktop\emotion_report.pdf"
        name = tracker.get_slot('name')
        dynamic_html = generate_dynamic_html(predicted_emotion,name)
        save_html_to_pdf(dynamic_html, pdf_filename)

        dispatcher.utter_message(text=f"Bye bye! If you ever need assistance, I'll be here. Also, here is a report on your emotional wellness:\n{pdf_filename}")

        return []
    
def generate_dynamic_html(predicted_emotion: str,name: str) -> str:
    # Define the dynamic HTML content
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
                <p>Predicted Emotion: {predicted_emotion}</p>
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
    
        
    
    
    
