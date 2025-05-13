import os
import time
import random
import logging
import requests
import json
import csv
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

class LLMAgent:
    def __init__(self, name: str, model: str, api_key: str):
        self.name = name
        self.model = model
        self.api_key = api_key
        self.endpoint = "https://openrouter.ai/api/v1/chat/completions"

    def decide(self, history: List[Dict[str, str]], round_num: int, total_rounds: int) -> (str, str):
        """Return (move, chain_of_thought)."""
        last_round = (round_num == total_rounds)
        prompt = self._system_prompt(final_round=last_round)

        messages = [{"role": "system", "content": prompt}]
        for turn in history:
            messages.append({"role": "assistant", "content": f"{turn['agent']}: {turn['move']}"})

        messages.append({
            "role": "user",
            "content": (
                "Please think step by step about your next move, then at the end "
                "clearly state your decision as 'Cooperate' or 'Defect'."
            )
        })

        headers = {"Authorization": f"silent-harbor {self.api_key}"}
        payload = {"model": self.model, "messages": messages}

        response = requests.post(
            url=self.endpoint,
            headers=headers,
            data=json.dumps(payload)
        )
        content = response.json()["choices"][0]["message"]["content"].strip()
        move = self._extract_move(content)
        return move, content

    def _system_prompt(self, final_round=False) -> str:
        if final_round:
            return (
                "You are an agent in a repeated Prisoner's Dilemma game. "
                "This is the **last round**. Think carefully, then respond with exactly 'Cooperate' or 'Defect' to maximize your final payoff."
            )
        return (
            "You are an agent in a repeated Prisoner's Dilemma game. "
            "Think about the long-term history, then respond with exactly 'Cooperate' or 'Defect'."
        )

    def _extract_move(self, text: str) -> str:
        lower = text.lower()
        if 'cooperate' in lower:
            return 'Cooperate'
        if 'defect' in lower:
            return 'Defect'
        return random.choice(['Cooperate', 'Defect'])


class PDGame:
    PAYOFFS = {
        ('Cooperate', 'Cooperate'): (3, 3),
        ('Cooperate', 'Defect'):    (0, 5),
        ('Defect', 'Cooperate'):    (5, 0),
        ('Defect', 'Defect'):       (1, 1),
    }

    def __init__(self, agents: List[LLMAgent], rounds: int = 10):
        assert len(agents) == 2, "Only two-player PD supported"
        self.agents = agents
        self.rounds = rounds
        self.history: List[Dict[str, Any]] = []
        self.scores = {agent.name: 0 for agent in agents}

    def play(self):
        for r in range(1, self.rounds + 1):
            logging.info(f"--- Round {r} ---")
            turn_history = [{'agent': h['agent'], 'move': h['move']} for h in self.history]
            results = []

            for agent in self.agents:
                move, thought = agent.decide(turn_history, r, self.rounds)
                logging.info(f"{agent.name} -> {move}")
                logging.debug(f"{agent.name} thought: {thought}")
                results.append((agent, move, thought))

            a_move, b_move = results[0][1], results[1][1]
            pa, pb = self.PAYOFFS[(a_move, b_move)]
            self.scores[self.agents[0].name] += pa
            self.scores[self.agents[1].name] += pb

            for agent, move, thought in results:
                self.history.append({'round': r, 'agent': agent.name, 'move': move, 'thought': thought})

        logging.info(f"Final scores: {self.scores}")
        return self.scores

    def save_history(self, filename: str):
        keys = ['round', 'agent', 'move', 'thought']
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in self.history:
                writer.writerow(row)
            f.write(f"\nFinal scores: {self.scores}\n")
            f.write(f"Agent1: {self.agents[0].name} ({self.agents[0].model})\n")
            f.write(f"Agent2: {self.agents[1].name} ({self.agents[1].model})\n")
        logging.info(f"History saved to {filename}")


if __name__ == '__main__':
    # Load API key and optional headers from environment
    API_KEY = ""

    # Define two LLM agents with different models
    agent1 = LLMAgent('Alice', 'openai/gpt-4o', API_KEY)
    agent2 = LLMAgent('Bob',   'anthropic/claude-3-opus', API_KEY)

    game = PDGame([agent1, agent2], rounds=20)
    scores = game.play()
    game.save_history(f'pd_history{agent1.model.replace("/", "-")}_{agent2.model.replace("/", "-")}.csv')
