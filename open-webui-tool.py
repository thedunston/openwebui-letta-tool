"""
title: Manage Letta Server and Agents
author: Duane Dunston (pair program with Deepseek)
author_url: https://github.com/thedunston
git_url: https://github.com/thedunston/openwebui-letta-tool.git
description: Manage and use Letta within Open Web UI
version: 0.0.2
licence: MIT
"""

"""
Current commands:

agent create AGENTNAME - Creates a new Agent.
agent list - List current agents.
agent send AGENTNAME MESSAGE - Sends a message to an Agent and returns the response.
agent archivemem AGENTNAME - Sends data to archival memory (support multilines).
agent clearhistory AGENTNAME - Clears the chat history (not memory).
gent help - This help screen.

The valve AGENT_API_BASE_URL allows specifying a different host where the Letta Server is located.

"""

import requests
import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
import asyncio


class Tools:
    class Valves(BaseModel):
        AGENT_API_BASE_URL: str = Field(
            default="http://localhost:8283",
            description="The base URL for the agent API.",
        )
        MAX_RETRIES: int = Field(
            default=3,
            description="Maximum number of retries for failed requests.",
        )
        TIMEOUT: int = Field(
            default=30,
            description="Timeout for the HTTP request in seconds.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.headers = {
            "Content-Type": "application/json",
        }

    async def _send_request(
        self,
        url: str,
        payload: dict,
        description: str,
        method: str = "POST",
    ) -> str:
        """
        Generic function to send a request with retries.
        """
        for attempt in range(self.valves.MAX_RETRIES):
            try:
                print(f"Attempt {attempt + 1}: {description}")
                if method.upper() == "GET":
                    response = requests.get(
                        url,
                        headers=self.headers,
                        timeout=self.valves.TIMEOUT,
                    )
                elif method.upper() == "PATCH":
                    response = requests.patch(
                        url,
                        headers=self.headers,
                        json=payload,
                        timeout=self.valves.TIMEOUT,
                    )
                else:  # Default to POST
                    response = requests.post(
                        url,
                        headers=self.headers,
                        json=payload,
                        timeout=self.valves.TIMEOUT,
                    )

                response.raise_for_status()

                print("Request successful:", description)
                return response.text

            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.valves.MAX_RETRIES - 1:
                    return json.dumps({"error": str(e)})

    async def list_agents(self) -> str:
        """
        List all agents and their IDs.
        """
        url = f"{self.valves.AGENT_API_BASE_URL}/v1/agents/"
        print("Fetching list of agents")

        response = requests.get(url, headers=self.headers)
        if response.status_code != 200:
            print("Failed to fetch agents:", response.text)
            return json.dumps({"error": response.text})

        agents = response.json()
        formatted_agents = "\n".join(
            f"{agent['name']}: {agent['id']}" for agent in agents
        )
        print("Agents listed successfully")
        return formatted_agents

    async def clear_history(self, user_input: str) -> str:
        """
        Clear the message history for a specific agent and verify the operation.
        """
        if not user_input.lower().startswith("agent clearhistory"):
            return json.dumps(
                {"error": "Invalid command. Use 'agent clearhistory AGENTNAME'."}
            )

        # Split the command.
        parts = user_input.split()
        if len(parts) < 3:
            return json.dumps(
                {"error": "Invalid command format. Use 'agent clearhistory AGENTNAME'."}
            )

        # 2 field in the index is the agent name.
        agent_name = parts[2]

        # Fetch the list of agents to resolve the agent ID
        agents_response = await self.list_agents()
        if agents_response.startswith("{"):
            return agents_response

        agents = {}
        for line in agents_response.split("\n"):
            name, agent_id = line.split(": ")
            agents[name] = agent_id

        if agent_name not in agents:
            return json.dumps({"error": f"Agent '{agent_name}' not found."})

        agent_id = agents[agent_name]

        # Step 1: Send PATCH request to reset messages
        reset_url = (
            f"{self.valves.AGENT_API_BASE_URL}/v1/agents/{agent_id}/reset-messages"
        )
        print(f"Clearing history for agent: {agent_name}")
        reset_response = await self._send_request(
            reset_url, {}, "clearing history", method="PATCH"
        )

        if reset_response.startswith("{"):
            return reset_response  # Return the error if the request failed

        # Step 2: Send GET request to verify the message count
        messages_url = f"{self.valves.AGENT_API_BASE_URL}/v1/agents/{agent_id}/messages"
        print(f"Verifying message count for agent: {agent_name}")
        messages_response = await self._send_request(
            messages_url, {}, "fetching messages", method="GET"
        )

        if messages_response.startswith("{"):
            return messages_response  # Return the error if the request failed

        # Parse the messages response
        messages_data = json.loads(messages_response)
        message_count = len(messages_data.get("messages", []))

        if message_count == 1:
            return f"Message history cleared successfully for agent '{agent_name}'. Current message count: {message_count}."
        else:
            return f"Failed to clear message history for agent '{agent_name}'. Current message count: {message_count}."

    async def send_message(
        self,
        user_input: str,
    ) -> str:
        """
        Send a message to a specific agent.
        """
        if not user_input.lower().startswith("agent send"):
            return json.dumps(
                {"error": "Invalid command. Use 'agent send AGENTNAME MESSAGE'."}
            )

        parts = user_input.split()
        if len(parts) < 4:
            return json.dumps(
                {"error": "Invalid command format. Use 'agent send AGENTNAME MESSAGE'."}
            )

        agent_name = parts[2]
        message = " ".join(parts[3:])

        # Fetch the list of agents to resolve the agent ID
        agents_response = await self.list_agents()
        if agents_response.startswith("{"):
            return agents_response

        agents = {}
        for line in agents_response.split("\n"):
            name, agent_id = line.split(": ")
            agents[name] = agent_id

        if agent_name not in agents:
            return json.dumps({"error": f"Agent '{agent_name}' not found."})

        agent_id = agents[agent_name]
        url = f"{self.valves.AGENT_API_BASE_URL}/v1/agents/{agent_id}/messages/stream"
        payload = {"messages": [{"role": "user", "content": message}]}

        print(f"Sending message to agent: {agent_name}")
        return await self._send_request(url, payload, "Sending message")

    async def send_archivemem(
        self,
        user_input: str,
    ) -> str:
        """
        Send archival memory to a specific agent.
        MEMORY includes everything after AGENTNAME, even if it contains newlines or special characters.
        """
        if not user_input.lower().startswith("agent archivemem"):
            return json.dumps(
                {"error": "Invalid command. Use 'agent archivemem AGENTNAME MEMORY'."}
            )

        parts = user_input.split(
            maxsplit=2
        )  # Split into 3 parts: ["agent", "archivemem", "AGENTNAME MEMORY"]
        if len(parts) < 3:
            return json.dumps(
                {
                    "error": "Invalid command format. Use 'agent archivemem AGENTNAME MEMORY'."
                }
            )

        agent_name = parts[2].split(maxsplit=1)[
            0
        ]  # Extract AGENTNAME (first word after "agent archivemem")
        memory = parts[2][
            len(agent_name) :
        ].strip()  # Extract MEMORY (everything after AGENTNAME)

        # Fetch the list of agents to resolve the agent ID
        agents_response = await self.list_agents()
        if agents_response.startswith("{"):
            return agents_response

        agents = {}
        for line in agents_response.split("\n"):
            name, agent_id = line.split(": ")
            agents[name] = agent_id

        if agent_name not in agents:
            return json.dumps({"error": f"Agent '{agent_name}' not found."})

        # Prepare the payload with JSON-encoded memory
        agent_id = agents[agent_name]
        url = f"{self.valves.AGENT_API_BASE_URL}/v1/agents/{agent_id}/archival-memory"
        payload = {
            "text": memory
        }  # JSON-encoding is handled automatically by requests.post

        print(f"Sending archival memory to agent: {agent_name}")
        response = await self._send_request(url, payload, "Sending archival memory")

        # Verify if the memory was stored successfully
        if response.startswith("{"):  # Check if the response is an error
            return response

        # Extract the text to match
        if len(memory) <= 50:
            text_to_match = memory
        else:
            text_to_match = memory[:50]  # Use the first 50 characters

        # Make a GET request to retrieve the archival memory
        get_url = (
            f"{self.valves.AGENT_API_BASE_URL}/v1/agents/{agent_id}/archival-memory"
        )
        try:
            get_response = requests.get(
                get_url, headers=self.headers, timeout=self.valves.TIMEOUT
            )
            get_response.raise_for_status()
            archival_memory = get_response.json()

            # Check if the text_to_match exists in the archival memory
            if any(text_to_match in entry.get("text", "") for entry in archival_memory):
                return json.dumps(
                    {"status": "Memory stored and verified successfully."}
                )
            else:
                return json.dumps(
                    {"error": "Memory was stored but could not be verified."}
                )
        except requests.exceptions.RequestException as e:
            return json.dumps({"error": f"Failed to verify memory: {str(e)}"})

        async def create_agent(self, user_input: str) -> str:
            """
            Create a new agent by parsing the user input and sending a POST request to the agent API.
            """
            if not user_input.lower().startswith("agent create"):
                return json.dumps(
                    {"error": "Invalid command. Use 'agent create AGENTNAME'."}
                )

            agent_name = user_input[len("agent create ") :].strip()
            if not agent_name:
                return json.dumps({"error": "Agent name cannot be empty."})

            url = f"{self.valves.AGENT_API_BASE_URL}/v1/agents/"
            payload = {
                "name": agent_name,
                "model": "letta/letta-free",
                "embedding": "letta/letta-free",
            }

            print(f"Creating agent: {agent_name}")
            result = await self._send_request(url, payload, "Creating agent")

            # Check if the agent was created successfully
            if result.startswith("{"):
                return result  # Return the error if the request failed

            # If successful, list all agents to confirm the new agent is added
            agents_list = await self.list_agents()
            return f"Agent '{agent_name}' created successfully.\nUpdated list of agents:\n{agents_list}"

    async def help_agent(self):
        return f"agent create AGENTNAME - Creates a new Agent.\nagent list - List current agents.\nagent send AGENTNAME MESSAGE - Sends a message to an Agent and returns the response.\nagent archivemem AGENTNAME - Sends data to archival memory (support multilines).\nagent clearhistory AGENTNAME - Clears the chat history (not memory).\nagent help - This help screen.\n"

    # Add new commands below in command_map.
    async def handle_command(
        self,
        user_input: str,
    ) -> str:
        """
        Parse the user input and route it to the appropriate function using a dictionary dispatch.
        """
        parts = user_input.lower().split()
        print(f"Parsed command: {parts}")  # Debug print

        if not parts:
            return json.dumps({"error": "No command provided."})

        # Ensure there are at least two words to form a command
        if len(parts) < 2:
            return json.dumps({"error": "Invalid command format."})

        # Extract the first two words as the command
        command = " ".join(parts[:2])

        # Dictionary mapping commands to their corresponding functions
        command_map = {
            "agent create": self.create_agent,
            "agent list": self.list_agents,
            "agent send": self.send_message,
            "agent archivemem": self.send_archivemem,
            "agent clearhistory": self.clear_history,
            "agent help": self.help_agent,
        }

        # Get the function from the command map
        if command in command_map:
            return await command_map[command](user_input)
        else:
            return json.dumps({"error": "Invalid command."})
