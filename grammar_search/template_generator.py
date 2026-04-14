#!/usr/bin/env python3
"""
Template-based code generator with text-based LLM infrastructure.
Each component uses Info objects with full text responses containing \\boxed{} for answers.
"""

from typing import List, Dict, Tuple, Optional

# Dataset-specific role mappings
DATASET_ROLE_MAPPINGS = {
    "math": {
        "singular": "Math Professor",
        "three": ["Math Professor", "Grade School Teacher", "Math Enthusiast"],
        "five": ["Math Professor", "Grade School Teacher", "Math Enthusiast", "Research Scientist", "Teaching Assistant"]
    },
    "gpqa": {
        "singular": "Science Professor",
        "three": ["Physics Expert", "Chemistry Expert", "Biology Expert"],
        "five": ["Science Professor", "Physics Expert", "Chemistry Expert", "Biology Expert", "Science Generalist"]
    },
    "aime": {
        "singular": "Competition Math Expert",
        "three": ["Competition Math Expert", "Problem Solver", "Math Olympian"],
        "five": ["Competition Math Expert", "Problem Solver", "Math Olympian", "Strategic Thinker", "Pattern Recognition Expert"]
    },
    "musique": {
        "singular": "Information Retrieval Expert",
        "three": ["Information Retrieval Expert", "Reasoning Expert", "Fact Checker"],
        "five": ["Information Retrieval Expert", "Reasoning Expert", "Fact Checker", "Research Analyst", "Knowledge Synthesizer"]
    },
    "mmlupro": {
        "singular": "Academic Expert",
        "three": ["Academic Expert", "Critical Analyst", "Subject Specialist"],
        "five": ["Academic Expert", "Critical Analyst", "Subject Specialist", "Research Scholar", "Problem Solver"]
    }
}

# Component templates with text-based infrastructure
ISOLATED_TEMPLATES = {
    "StepByStepReasoner_singular": """
def component_{idx}_step_by_step_singular(taskInfo, prev_answer=None):
    '''StepByStepReasoner: Chain-of-Thought reasoning with single agent'''
    inputs = [taskInfo]
    instruction = "Please think step by step and then solve the task. Put your final answer in \\\\boxed{}."
    
    if prev_answer is not None:
        inputs.append(prev_answer)
        instruction = "Based on the previous solution above, please think step by step and provide your own solution. Put your final answer in \\\\boxed{}."
    
    agent = LLMAgentBase('Chain-of-Thought Agent')
    answer = agent(inputs, instruction)
    return answer, agent  # Return agent for potential SelfCriticIteration
""",
    
    "StepByStepReasoner_plural": """
def component_{idx}_step_by_step_plural(taskInfo, prev_answer=None):
    '''Multiple parallel step-by-step reasoners'''
    inputs = [taskInfo]
    instruction = "Please think step by step and then solve the task. Put your final answer in \\\\boxed{}."
    
    if prev_answer is not None:
        inputs.append(prev_answer)
        instruction = "Based on the previous solution above, please think step by step and provide your own solution. Put your final answer in \\\\boxed{}."
    
    N = {count}
    agents = [LLMAgentBase('Chain-of-Thought Agent', temperature=0.8) for _ in range(N)]
    
    all_results = []
    for i in range(N):
        answer = agents[i](inputs, instruction)
        all_results.append(answer)
    
    return all_results, agents
""",
    
    "RoleBasedReasoner_singular": """
def component_{idx}_role_based_singular(taskInfo, prev_answer=None):
    '''RoleBasedReasoner: Single role-based agent'''
    inputs = [taskInfo]
    instruction = "Please think step by step and then solve the task. Put your final answer in \\\\boxed{}."
    
    if prev_answer is not None:
        inputs.append(prev_answer)
        instruction = "Based on the previous solution above, please think step by step from your role perspective and provide your own solution. Put your final answer in \\\\boxed{}."
    
    role = '{role}'
    agent = LLMAgentBase('Role-Based Reasoner', role=role)
    answer = agent(inputs, instruction)
    return answer, agent
""",
    
    "RoleBasedReasoner_plural": """
def component_{idx}_role_based_plural(taskInfo, prev_answer=None):
    '''Multiple role-based agents with different perspectives'''
    inputs = [taskInfo]
    instruction = "Please think step by step and then solve the task. Put your final answer in \\\\boxed{}."
    
    if prev_answer is not None:
        inputs.append(prev_answer)
        instruction = "Based on the previous solution above, please think step by step from your role perspective and provide your own solution. Put your final answer in \\\\boxed{}."
    
    roles = {roles}
    agents = [LLMAgentBase('Role-Based Agent', temperature=0.8, role=role) for role in roles]
    
    all_results = []
    for agent in agents:
        answer = agent(inputs, instruction)
        all_results.append(answer)
    
    return all_results, agents
""",
    
    "SelfCriticIteration_singular": """
def component_{idx}_self_critic_singular(taskInfo, answer, agent=None):
    '''Pure refinement iteration - single round'''
    if agent is None:
        # Create a fresh agent for refinement when none is provided
        agent = LLMAgentBase('Refinement Agent')
    
    inputs = [taskInfo]
    
    critic_instruction = "Please review the answer above and provide detailed feedback on any errors or improvements needed. "
    critic_instruction += "IMPORTANT: At the end of your feedback, you must indicate whether the answer is correct by writing either [CORRECT] or [INCORRECT]. "
    critic_instruction += "Write [CORRECT] ONLY if you are absolutely certain the answer is completely correct. Otherwise write [INCORRECT]."
    
    critic_agent = LLMAgentBase('Critic Agent')
    
    reflect_instruction = "Given the previous attempt and feedback, carefully consider where you could go wrong. Using insights from the feedback, try to solve the task better. Put your final answer in \\\\boxed{}."
    
    feedback = critic_agent([taskInfo, answer], critic_instruction, 0)
    if '[CORRECT]' not in feedback.content:
        inputs.extend([answer, feedback])
        answer = agent(inputs, reflect_instruction, 1)
    
    return answer
""",
    
    "SelfCriticIteration_plural": """
def component_{idx}_self_critic_plural(taskInfo, answer, agent=None):
    '''Pure refinement iteration - multiple rounds'''
    if agent is None:
        # Create a fresh agent for refinement when none is provided
        agent = LLMAgentBase('Refinement Agent')
    
    inputs = [taskInfo]
    
    critic_instruction = "Please review the answer above and provide detailed feedback on any errors or improvements needed. "
    critic_instruction += "IMPORTANT: At the end of your feedback, you must indicate whether the answer is correct by writing either [CORRECT] or [INCORRECT]. "
    critic_instruction += "Write [CORRECT] ONLY if you are absolutely certain the answer is completely correct. Otherwise write [INCORRECT]."
    
    critic_agent = LLMAgentBase('Critic Agent')
    
    reflect_instruction = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better. Put your final answer in \\\\boxed{}."
    
    N_max = {rounds}
    for i in range(N_max):
        feedback = critic_agent([taskInfo, answer], critic_instruction, i)
        if '[CORRECT]' in feedback.content:
            break
        inputs.extend([answer, feedback])
        answer = agent(inputs, reflect_instruction, i + 1)
    
    return answer
""",
    
    "MajorityVoter": """
def component_{idx}_majority_voter(taskInfo, all_results):
    '''Apply semantic-aware majority voting using LLM'''
    voting_instruction = "Given these " + str(len(all_results)) + " solutions to the same problem:\\n\\n"
    for i, response in enumerate(all_results, 1):
        voting_instruction += "\\nSolution " + str(i) + ":\\n" + response.content + "\\n"
    
    voting_instruction += "\\nAnalyze these solutions and identify which answer appears most frequently (considering equivalent answers as the same).\\n"
    voting_instruction += "Then, select ONE solution from above that contains the most common answer.\\n"
    voting_instruction += "Copy that ENTIRE solution verbatim, exactly as it appears above, including all its reasoning steps.\\n"
    voting_instruction += "Do not summarize or paraphrase - reproduce the complete original solution text.\\n"
    voting_instruction += "After copying the solution, ensure your final answer is in \\\\boxed{}."
    
    voting_agent = LLMAgentBase('Voting Agent', temperature=0.1)
    final_answer = voting_agent([taskInfo], voting_instruction)
    
    return final_answer
""",
    
    "ConsensusBuilder": """
def component_{idx}_consensus_builder(taskInfo, all_results):
    '''Final decision-making by synthesizing all solutions'''
    final_instruction = "Given all the above solutions, analyze them carefully and provide a final solution and answer. Put your final answer in \\\\boxed{}."
    final_agent = LLMAgentBase('Final Decision Agent', temperature=0.1)
    
    answer = final_agent([taskInfo] + all_results, final_instruction)
    return answer
""",
    
    "DebateIteration_singular": """
def component_{idx}_debate_singular(taskInfo, all_results, agents):
    '''Single iteration round where agents consider each other's solutions'''
    iteration_instruction = "Given solutions to the problem from all agents (including yourself), consider all perspectives and provide an updated solution and answer. Put your final answer in \\\\boxed{}."
    
    current_results = []
    for i in range(len(agents)):
        answer = agents[i]([taskInfo] + all_results, iteration_instruction)
        current_results.append(answer)
    
    return current_results, agents
""",
    
    "DebateIteration_plural": """
def component_{idx}_debate_plural(taskInfo, all_results, agents):
    '''Pure iteration loop where agents debate over multiple rounds'''
    iteration_instruction = "Given solutions to the problem from all agents (including yourself), consider all perspectives and provide an updated solution and answer. Put your final answer in \\\\boxed{}."
    
    N_max = {rounds}
    for round_num in range(N_max):
        current_results = []
        for i in range(len(agents)):
            answer = agents[i]([taskInfo] + all_results, iteration_instruction)
            current_results.append(answer)
        
        all_results = current_results
    
    return all_results, agents
""",
    
    "MultiSelfCriticIteration_singular": """
def component_{idx}_multi_self_critic_singular(taskInfo, all_results, agents):
    '''Each agent independently refines through self-criticism'''
    multi_agent_inputs = [[taskInfo] for _ in range(len(agents))]
    
    critic_instruction = "Please review the answer above and provide detailed feedback on any errors or improvements needed. "
    critic_instruction += "IMPORTANT: At the end of your feedback, you must indicate whether the answer is correct by writing either [CORRECT] or [INCORRECT]. "
    critic_instruction += "Write [CORRECT] ONLY if you are absolutely certain the answer is completely correct. Otherwise write [INCORRECT]."
    
    reflect_instruction = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better. Put your final answer in \\\\boxed{}."
    
    critics = [LLMAgentBase('Critic for Agent ' + str(i)) for i in range(len(agents))]
    agents_done = [False] * len(agents)
    
    refined_results = []
    for i in range(len(agents)):
        agent_answer = all_results[i]
        
        feedback = critics[i]([taskInfo, agent_answer], critic_instruction, 0)
        
        if '[CORRECT]' not in feedback.content:
            multi_agent_inputs[i].extend([agent_answer, feedback])
            new_answer = agents[i](multi_agent_inputs[i], reflect_instruction, 1)
            refined_results.append(new_answer)
        else:
            agents_done[i] = True
            refined_results.append(agent_answer)
    
    return refined_results, agents
""",
    
    "MultiSelfCriticIteration_plural": """
def component_{idx}_multi_self_critic_plural(taskInfo, all_results, agents):
    '''Each agent independently refines through self-criticism over multiple rounds'''
    multi_agent_inputs = [[taskInfo] for _ in range(len(agents))]
    
    critic_instruction = "Please review the answer above and provide detailed feedback on any errors or improvements needed. "
    critic_instruction += "IMPORTANT: At the end of your feedback, you must indicate whether the answer is correct by writing either [CORRECT] or [INCORRECT]. "
    critic_instruction += "Write [CORRECT] ONLY if you are absolutely certain the answer is completely correct. Otherwise write [INCORRECT]."
    
    reflect_instruction = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better. Put your final answer in \\\\boxed{}."
    
    critics = [LLMAgentBase('Critic for Agent ' + str(i)) for i in range(len(agents))]
    agents_done = [False] * len(agents)
    
    N_max = {rounds}
    for round_num in range(N_max):
        refined_results = []
        
        for i in range(len(agents)):
            agent_answer = all_results[i]
            
            if agents_done[i]:
                refined_results.append(agent_answer)
            else:
                feedback = critics[i]([taskInfo, agent_answer], critic_instruction, round_num)
                
                if '[CORRECT]' not in feedback.content:
                    multi_agent_inputs[i].extend([agent_answer, feedback])
                    new_answer = agents[i](multi_agent_inputs[i], reflect_instruction, round_num + 1)
                    refined_results.append(new_answer)
                else:
                    agents_done[i] = True
                    refined_results.append(agent_answer)
        
        all_results = refined_results
        
        if all(agents_done):
            break
    
    return all_results, agents
"""
}


class IsolatedTemplateGenerator:
    """Generate code with isolated execution contexts for each component using text-based infrastructure."""
    
    def __init__(self, dataset_type: Optional[str] = None):
        """
        Initialize template generator with dataset type.
        
        Args:
            dataset_type: Type of dataset ('math', 'gpqa', 'aime', 'musique').
                         Required parameter.
        
        Raises:
            ValueError: If dataset_type is not provided or not supported.
        """
        if dataset_type is None:
            raise ValueError("dataset_type is required. Must be one of: 'math', 'gpqa', 'aime', 'musique'")
        
        if dataset_type not in DATASET_ROLE_MAPPINGS:
            raise ValueError(f"Unsupported dataset_type: '{dataset_type}'. Must be one of: {list(DATASET_ROLE_MAPPINGS.keys())}")
        
        self.dataset_type = dataset_type
    
    def parse_component(self, component: str) -> Tuple[str, str, int]:
        """Parse component string to extract name, type, and parameter value."""
        base_name = component.split('(')[0]
        param_type = None
        param_value = 1
        
        if '(' in component:
            param_str = component.split('(')[1].rstrip(')')
            if 'count=' in param_str:
                param_type = 'count'
                param_value = int(param_str.split('=')[1])
            elif 'rounds=' in param_str:
                param_type = 'rounds'
                param_value = int(param_str.split('=')[1])
        
        return base_name, param_type, param_value
    
    def get_template_key(self, base_name: str, param_value: int) -> str:
        """Get the template key for a component."""
        if base_name in ['MajorityVoter', 'ConsensusBuilder']:
            return base_name
        
        if param_value == 1:
            return f"{base_name}_singular"
        else:
            return f"{base_name}_plural"
    
    def _get_roles_for_dataset(self, count: int) -> any:
        """
        Get appropriate roles based on dataset type and count.
        
        Args:
            count: Number of roles needed (1, 3, or 5)
            
        Returns:
            Single role string for count=1, list of roles for count>1
        """
        # Get dataset-specific roles
        dataset_roles = DATASET_ROLE_MAPPINGS[self.dataset_type]
        
        if count == 1:
            return dataset_roles['singular']
        elif count == 3:
            return dataset_roles['three']
        elif count == 5:
            return dataset_roles['five']
        else:
            # For other counts, use the first N roles from the five list
            five_roles = dataset_roles['five']
            if count <= len(five_roles):
                return five_roles[:count]
            else:
                # If requested count exceeds available roles, repeat some
                return (five_roles * ((count // len(five_roles)) + 1))[:count]
    
    def generate_code(self, component_sequence: List[str], system_name: str = "GeneratedSystem") -> Dict[str, str]:
        """Generate complete forward function with isolated execution contexts."""
        function_defs = []
        orchestration_calls = []
        
        # Track state variables for passing between components
        state_tracking = {
            'has_answer': False,  # Single Info object
            'has_all_results': False,  # List of Info objects
            'has_agents': False,  # List of agent references
            'has_agent': False,  # Single agent reference
            'last_single_agent_idx': None
        }
        
        for idx, component in enumerate(component_sequence):
            base_name, param_type, param_value = self.parse_component(component)
            template_key = self.get_template_key(base_name, param_value)
            
            if template_key not in ISOLATED_TEMPLATES:
                continue
            
            # Get template
            template = ISOLATED_TEMPLATES[template_key]
            
            # Basic substitutions
            template = template.replace('{idx}', str(idx))
            if param_type == 'count':
                template = template.replace('{count}', str(param_value))
            elif param_type == 'rounds':
                template = template.replace('{rounds}', str(param_value))
            
            # Handle role-based reasoner templates
            if base_name == 'RoleBasedReasoner':
                if param_value == 1:
                    role = self._get_roles_for_dataset(1)
                    template = template.replace('{role}', role)
                else:
                    # For any plural count, use the appropriate roles
                    roles = self._get_roles_for_dataset(param_value)
                    template = template.replace('{roles}', str(roles))
            
            function_defs.append(template)
            
            # Generate orchestration call based on component type
            call = self._generate_orchestration_call(
                idx, base_name, param_value, state_tracking
            )
            orchestration_calls.append(call)
        
        # Build complete code
        code = "def forward(self, taskInfo):\n"
        code += "    # Component function definitions\n"
        for func_def in function_defs:
            code += "    " + "\n    ".join(func_def.splitlines()) + "\n\n"
        
        code += "    # Orchestration\n"
        for call in orchestration_calls:
            code += f"    {call}\n"
        
        code += "    return answer"
        
        return {
            "name": system_name,
            "code": code,
            "analysis": f"Generated with isolated contexts from: {component_sequence}",
            "dataset_type": self.dataset_type
        }
    
    def _generate_orchestration_call(self, idx: int, base_name: str, param_value: int, state: Dict) -> str:
        """Generate the orchestration call for a component based on state tracking."""
        
        if base_name in ['StepByStepReasoner', 'RoleBasedReasoner']:
            if param_value == 1:  # Singular
                if state['has_answer']:
                    call = f"answer, agent = component_{idx}_{self._get_func_name(base_name, param_value)}(taskInfo, answer)"
                else:
                    call = f"answer, agent = component_{idx}_{self._get_func_name(base_name, param_value)}(taskInfo)"
                state['has_answer'] = True
                state['has_agent'] = True
                state['last_single_agent_idx'] = idx
                state['has_all_results'] = False
                state['has_agents'] = False
            else:  # Plural
                if state['has_answer']:
                    call = f"all_results, agents = component_{idx}_{self._get_func_name(base_name, param_value)}(taskInfo, answer)"
                else:
                    call = f"all_results, agents = component_{idx}_{self._get_func_name(base_name, param_value)}(taskInfo)"
                state['has_all_results'] = True
                state['has_agents'] = True
                state['has_answer'] = False
                state['has_agent'] = False
        
        elif base_name == 'SelfCriticIteration':
            if state['has_agent']:
                call = f"answer = component_{idx}_self_critic_{'singular' if param_value == 1 else 'plural'}(taskInfo, answer, agent)"
            else:
                # Pass None for agent - the function will create its own
                call = f"answer = component_{idx}_self_critic_{'singular' if param_value == 1 else 'plural'}(taskInfo, answer, None)"
            state['has_answer'] = True
            state['has_agent'] = False  # SelfCritic doesn't return agent
        
        elif base_name in ['MajorityVoter', 'ConsensusBuilder']:
            call = f"answer = component_{idx}_{self._get_func_name(base_name, param_value)}(taskInfo, all_results)"
            state['has_answer'] = True
            state['has_all_results'] = False
            state['has_agents'] = False
            state['has_agent'] = False
        
        elif base_name in ['DebateIteration', 'MultiSelfCriticIteration']:
            call = f"all_results, agents = component_{idx}_{self._get_func_name(base_name, param_value)}(taskInfo, all_results, agents)"
            state['has_all_results'] = True
            state['has_agents'] = True
            state['has_answer'] = False
            state['has_agent'] = False
        
        else:
            call = f"# Unknown component: {base_name}"
        
        return call
    
    def _get_func_name(self, base_name: str, param_value: int) -> str:
        """Get function name suffix for a component."""
        name_map = {
            'StepByStepReasoner': 'step_by_step',
            'RoleBasedReasoner': 'role_based',
            'SelfCriticIteration': 'self_critic',
            'MajorityVoter': 'majority_voter',
            'ConsensusBuilder': 'consensus_builder',
            'DebateIteration': 'debate',
            'MultiSelfCriticIteration': 'multi_self_critic'
        }
        
        base = name_map.get(base_name, base_name.lower())
        
        if base_name in ['MajorityVoter', 'ConsensusBuilder']:
            return base
        elif param_value == 1:
            return f"{base}_singular"
        else:
            return f"{base}_plural"


# Example usage
if __name__ == "__main__":
    # Test with different datasets
    for dataset in ['math', 'gpqa', 'aime', 'musique']:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print(f"{'='*60}")
        
        generator = IsolatedTemplateGenerator(dataset_type=dataset)
        
        # Test sequence with various components
        sequence = [
            'StepByStepReasoner(count=1)',
            'SelfCriticIteration(rounds=5)',
            'RoleBasedReasoner(count=1)',
            'MajorityVoter',
            'RoleBasedReasoner(count=5)',
            'DebateIteration(rounds=2)',
            'MultiSelfCriticIteration(rounds=5)',
            'ConsensusBuilder'
        ]
        
        result = generator.generate_code(sequence, f"{dataset.upper()}_System")
        
        print("Generated Code Preview:")
        print(result['code'])
        print(f"\nDataset type: {result['dataset_type']}")
        print(f"Analysis: {result['analysis']}")
