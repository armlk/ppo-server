from flask import Flask, request
from ppo_torch import Agent
# import numpy as np


# best_score = float("-inf")
# score_history = []
app = Flask(__name__)


@app.post("/action")
def action_handler():
    global agent
    inp = request.get_json().get("input")
    return list(agent.choose_action(inp))


@app.post("/reset-agent")
def reset_agent_handler():
    global agent
    print({**request.get_json()})
    agent = Agent(**request.get_json())
    return str(agent is not None)


@app.post("/learn")
def learn_handler():
    global agent
    json = request.get_json()

    for ts in json:
        agent.remember(**ts)
        # score = json.get("reward")
        # global score_history
        # score_history.append(score)
        # avg_score = np.mean(score_history[-100:])
        #
        # global best_score
        # if avg_score > best_score:
        #     best_score = avg_score
        #     agent.save_models()

    agent.learn()

    return ""


@app.post("/save-model")
def save_model_handler():
    global agent
    agent.save_models(request.get_json("name"))
    return ""


@app.post("/load-model")
def load_model_handler():
    global agent
    agent.load_models(request.get_json("name"))
    return ""
