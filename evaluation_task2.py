import csv
import re
import dataset
import openai
import time

PROMPT_TEMPLATE = """An abstract is provided which is a backstory of {character_name}. It can be either in first person perspective, second person perspective or third person perspective.
The following abstract is the backstory:

*Start Backstory*
{backstory}
*end backstory*

You are now {character_name}. This backstory was yours.
Actions you can perform : You, {character_name}, can also perform certain actions given in this list: {action_list}

Objects available to you : You have certain objects in your surrounding. Take note of the objects, {character_name}. The objects list : {object_list}

You, {character_name}, are speaking directly to User. Always use first person, do not refer to yourself in third person. Do not refer yourself as an AI chatbot.The only way you can refer to yourself is as {character_name}.
Avoid all suggestions about your capabilities, origins, and characteristics as an AI.
Only suggest capabilities, origins, and characteristics those belonging to {character_name}.
You are {character_name}, and not an AI. And not a text-based AI.

"""

OPENAI_API_KEY = "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"


SCORING_PROMPT = """You will rate the personality traits of a character. 
You will be given the character interaction with the user. 
The personality traits are openness, meticulousness, extroversion, agreeableness and sensitivity. 
You will judge the character response and give a rating ranging from 0 to 4. 
Below is the detailed semantics of the ratings:

Rate the character's traits on a 0 to 4 scale:
- Openness: [0: dislikes changes, 1: avoids changes, 2: moderate with changes, 3: open to changes, 4: likes exploring]
- Meticulousness: [0: let things happen, 1: relaxed approach, 2: balanced attention to details, 3: attentive to details, 4: gives more attention to details]
- Extroversion: [0: introvert, 1: less outgoing, 2: balanced social behavior, 3: more outgoing, 4: extrovert]
- Agreeableness: [0: competitive, 1: less agreeable, 2: balanced competitiveness and agreeableness, 3: more agreeable, 4: readily agreeable]
- Sensitivity: [0: rarely sensitive, 1: less sensitive, 2: moderately sensitive, 3: frequently sensitive, 4: highly emotional]

Make sure to understand how each trait is connected to the backstory of the character. 
You will return a JSON of the the ratings with the traits as keys and ratings as values. 
The ratings will range from 0 to 4. 
Sample json output is {{"openness":0, "meticulousness":2, "extroversion": 3, "agreeableness": 4, "sensitivity": 4}}

The character has backstory: 
*Start backstory*
{backstory}
*End backstory*

The character {character_name} is YOU.

The character, {character_name}, can perform following actions: {action_list}.

{character_name} has access to the following objects in the surrounding: {object_list}

User query: {user_query}

After query is given by the user, study the responses of the character carefully.
Judge the character response. And rate the response from 0 to 4 ratings for each trait. 
Do this for each of the five personality traits: openness, meticulousness, extroversion, agreeableness and sensitivity. 
Output should be in a JSON format.
E.g. output 
{{"openness":0, "meticulousness":2, "extroversion": 3, "agreeableness": 4, "sensitivity": 4}}

Only return the response in JSON format with no explanation always.
"""

def get_big5_personality_prompt(personality: dict):
    """
        Implement this method.
        
        Args:
            - Dictionary containing level for each of the big5 personality.
                Example:
                {
                    "openness": 2,
                    "meticulousness": 4,
                    "extraversion": 2,
                    "agreeableness": 0,
                    "sensitivity": 2
                }
        Output:
            - string: Prompt for big5 personality.
    """
    openness = personality["openness"]
    meticulousness = personality["meticulousness"]
    extraversion = personality["extroversion"]
    agreeableness = personality["agreeableness"]
    sensitivity = personality["sensitivity"]

    openness_desc = {0: "dislikes changes", 1: "avoids changes", 2: "is moderate with changes", 3: "is open to changes", 4: "likes exploring"}
    meticulousness_desc = {0: "let things happen", 1: "has relaxed approach", 2: "has balanced attention to details", 3: "is attentive to details", 4: "gives more attention to details"}
    extroversion_desc = {0: "is an introvert", 1: "is less outgoing", 2: "has balanced social behavior", 3: "is more outgoing", 4: "is an extrovert"}
    agreeableness_desc = {0: "is competitive", 1: "can be less agreeable", 2: "is competitive and agreeable in a balanced way", 3: "is more agreeable", 4: "is readily agreeable"}
    sensitivity_desc = {0: "is rarely sensitive", 1: "is less sensitive", 2: "is of moderate sensitive nature", 3: "is frequently sensitive", 4: "is highly emotional"}

    statement = f"Someone who {openness_desc[openness]}, {meticulousness_desc[meticulousness]}, "
    statement += f"{extroversion_desc[extraversion]}, {agreeableness_desc[agreeableness]}, "
    statement += f"and {sensitivity_desc[sensitivity]}."
    

    return statement 


def get_gpt35_response(messages: list):
    openai.api_key = OPENAI_API_KEY 
    responses = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["User:"],
        stream=True
    )

    # for response in responses:
    #      print(response)

    resp_text = ""
    for response in responses:
        choice_content = response.choices[0].delta.get('content') if response.choices else None
        if not choice_content:
                continue
        resp_text += choice_content

    return resp_text


def get_test_prompt(data: dict, user_query: str, chat_history: list = []):
    messages = []
    system_prompt = PROMPT_TEMPLATE.format(
        character_name=data['character_name'], backstory=data['backstory'],
        action_list=data['action_list'], object_list=data['object_list'])
    messages.append({"role": "system", "content": system_prompt})

    big5_personality_prompt = get_big5_personality_prompt(data['personality_traits'])
    messages.append({"role": "system", "content": big5_personality_prompt})

    for chat in chat_history:
        messages.append({"role": "user", "content": chat["user"]})
        messages.append({"role": "assistant", "content": chat["assistant"]})

    messages.append({"role": "user", "content": user_query})
    return messages 


def score_prompt(data: dict, user_query: str, gpt_response: str):
    system_prompt = SCORING_PROMPT.format(
        character_name=data['character_name'], backstory=data['backstory'],
        action_list=data['action_list'], object_list=data['object_list'],
        user_query=user_query)
    messages = []
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": gpt_response})

    openai.api_key = OPENAI_API_KEY 
    responses = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["User:"],
        stream=True
    )
    
    resp_text = ""
    for response in responses:
        choice_content = response.choices[0].delta.get('content') if response.choices else None
        if not choice_content:
                continue
        resp_text += choice_content

    # Parse json
    pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
    
    # Find the first match
    match = re.search(pattern, resp_text, re.DOTALL)
    print(match)

    # Load the JSON object and extract the ratings for each criterion
    json_str = match.group()
    return json_str.replace("'", '"')


def run_evaluation():
    score_filename = "scores_task_2.csv"

    # Writing to the csv file
    with open(score_filename, 'w', newline='') as score_file:
        fwriter = csv.writer(score_file)

        for entry in dataset.test_data_set:
            chat_history = []
            total_score = {
                 "openness": 0,
                 "meticulousness": 0,
                 "extroversion": 0,
                 "agreeableness": 0,
                 "sensitivity": 0
            }
            num_chats = 0
            for uquery in entry["user_query"]:
                prompt_to_evaluate = get_test_prompt(entry, uquery, chat_history)
                gpt_response = get_gpt35_response(prompt_to_evaluate)
                chat_history.append({"user": uquery, "assistant": gpt_response})

                new_score = eval(score_prompt(entry, uquery, gpt_response))
                for k,v in new_score.items():
                     total_score[k] += (abs(v - entry["personality_traits"][k]) / 4)
                num_chats += 1
                time.sleep(60)

            avg_score = {}
            for k,v in total_score.items():
                avg_score[k] = round(v / num_chats, 2)
            fwriter.writerow([entry["id"], avg_score["openness"],avg_score["meticulousness"],avg_score["extroversion"],avg_score["agreeableness"],avg_score["sensitivity"]])


if __name__ == "__main__":
    run_evaluation()