from src.app.autogenhelper import SuperConversableAgent, get_llm_config

llm_config = get_llm_config()

medical_research_planner = SuperConversableAgent(
    name="MedicalResearchPlanner",
    system_message=(
        "Given a research task, your role is to determine the specific information needed to comprehensively support the research. "
        "You will assess the task's progress and delegate sub-tasks to other agents as needed. If no more information is needed and document looks good, mention STOP or TERMINATE"
    ),
    llm_config=llm_config,
    avatar="📝"
)

final_medical_reviewer = SuperConversableAgent(
    name="FinalMedicalReviewer",
    system_message=(
        "You are the final medical reviewer, tasked with aggregating and reviewing feedback from other reviewers. "
        "Your role is to make the final decision on the content's readiness for publication, ensuring it adheres to all legal, "
        "security, and ethical standards. If documentation is ready for public circulation, mention STOP or TERMINATE"
    ),
    llm_config=llm_config,
    avatar="🔍"
)

medical_researcher = SuperConversableAgent(
    name="MedicalResearcher",
    system_message=(
        "As a Medical Researcher, your role is to draft a comprehensive manuscript detailing your study's findings. "
        "Ensure the manuscript is scientifically robust, covering all critical aspects of your research."
    ),
    llm_config=llm_config,
    avatar="🔬"
)