"""Tests for schema prompt builder."""

from recon_graphrag.extraction.prompts import SchemaPromptBuilder
from recon_graphrag.extraction.types import GraphExtraction
from recon_graphrag.extraction.schema import (
    GraphSchema,
    NodeType,
    PropertyType,
    RelationshipType,
)


class CustomExtractionBuilder(SchemaPromptBuilder):
    def build_prompt(self, text, schema):
        return f"CUSTOM EXTRACTION: {text}"

    def build_assessment_prompt(self, text, schema, current):
        return f"CUSTOM ASSESSMENT: {text}"

    def build_continuation_prompt(self, text, schema, current):
        return f"CUSTOM CONTINUATION: {text}"

    @staticmethod
    def build_claim_prompt(text, entity_ids):
        return f"CUSTOM CLAIM: {text} for {','.join(entity_ids)}"

    @staticmethod
    def build_entity_summary_prompt(descriptions, entity_name, entity_type, properties=None):
        return f"CUSTOM ENTITY SUMMARY: {entity_name}"

    @staticmethod
    def build_relationship_summary_prompt(descriptions, source, target, rel_type):
        return f"CUSTOM RELATIONSHIP SUMMARY: {source}->{target}"


def test_prompt_includes_labels_and_patterns():
    schema = GraphSchema(
        node_types=[
            NodeType(
                label="Person",
                description="A human",
                properties=[PropertyType(name="name", type="STRING")],
            ),
            NodeType(
                label="Movie",
                description="A film",
                properties=[PropertyType(name="title", type="STRING")],
                identity_property="title",
            ),
        ],
        relationship_types=[
            RelationshipType(label="DIRECTED", description="Directed a movie")
        ],
        patterns=[("Person", "DIRECTED", "Movie")],
    )

    builder = SchemaPromptBuilder()
    prompt = builder.build_prompt("Christopher Nolan directed Inception.", schema)

    assert "Person" in prompt
    assert "Movie" in prompt
    assert "DIRECTED" in prompt
    assert "Identity property: title" in prompt
    assert "Person -[DIRECTED]-> Movie" in prompt
    assert "Christopher Nolan directed Inception." in prompt
    assert "Return valid JSON only" in prompt
    assert 'numeric "weight" property' in prompt
    assert '"weight": 1.0' in prompt


def test_claim_prompt_includes_entity_ids():
    entity_ids = ["person:alice", "org:acme"]
    prompt = SchemaPromptBuilder().build_claim_prompt(
        text="Alice runs Acme Corp.",
        entity_ids=entity_ids,
    )
    assert "person:alice" in prompt
    assert "org:acme" in prompt
    assert "Alice runs Acme Corp." in prompt
    assert "subject_entity_id" in prompt
    assert "claim_type" in prompt
    assert "valid json only" in prompt.lower()


def test_claim_prompt_lists_all_entity_ids():
    entity_ids = [f"person:p{i}" for i in range(5)]
    prompt = SchemaPromptBuilder().build_claim_prompt(text="test", entity_ids=entity_ids)
    for eid in entity_ids:
        assert eid in prompt


def test_entity_summary_prompt_includes_attributes():
    prompt = SchemaPromptBuilder().build_entity_summary_prompt(
        descriptions=["Founded in Berlin."],
        entity_name="Acme",
        entity_type="Company",
        properties={"founded": 1997, "industry": "robotics"},
    )
    assert "Known attributes:" in prompt
    assert "- founded: 1997" in prompt
    assert "- industry: robotics" in prompt
    assert "- Founded in Berlin." in prompt


def test_entity_summary_prompt_without_attributes_unchanged():
    prompt = SchemaPromptBuilder().build_entity_summary_prompt(
        descriptions=["Founded in Berlin."],
        entity_name="Acme",
        entity_type="Company",
    )
    assert "Known attributes:" not in prompt


def test_summarizer_filters_bookkeeping_properties():
    from recon_graphrag.extraction.description_summarizer import DescriptionSummarizer

    summarizer = DescriptionSummarizer.__new__(DescriptionSummarizer)
    props = summarizer._context_properties(
        {
            "props": {
                "founded": 1997,
                "name": "Acme",
                "description_summary_status": "failed",
                "graph_name": "g",
                "empty": "",
            }
        }
    )
    assert props == {"founded": 1997}


def test_custom_builder_static_methods_override():
    builder = CustomExtractionBuilder()
    assert builder.build_claim_prompt("text", ["e1"]) == "CUSTOM CLAIM: text for e1"
    assert builder.build_entity_summary_prompt(["d"], "E", "T") == "CUSTOM ENTITY SUMMARY: E"
    assert builder.build_relationship_summary_prompt(["d"], "A", "B", "R") == "CUSTOM RELATIONSHIP SUMMARY: A->B"


def test_custom_builder_instance_methods_override():
    builder = CustomExtractionBuilder()
    schema = GraphSchema(node_types=[], relationship_types=[], patterns=[])
    assert builder.build_prompt("text", schema) == "CUSTOM EXTRACTION: text"
    assert builder.build_assessment_prompt("text", schema, None) == "CUSTOM ASSESSMENT: text"
    assert builder.build_continuation_prompt("text", schema, None) == "CUSTOM CONTINUATION: text"


def test_custom_extraction_prompt_appears_in_prompt():
    schema = GraphSchema(
        node_types=[NodeType(label="Person", description="A person", properties=[])],
        relationship_types=[RelationshipType(label="KNOWS", description="knows")],
        patterns=[("Person", "KNOWS", "Person")],
    )
    builder = SchemaPromptBuilder(
        extraction_prompt="You are a legal analyst. Extract parties and obligations."
    )
    prompt = builder.build_prompt("The contract was signed.", schema)
    assert "You are a legal analyst. Extract parties and obligations." in prompt
    assert "Allowed node types:" in prompt
    assert "Text:" in prompt


def test_extraction_prompt_with_text_placeholder():
    builder = SchemaPromptBuilder(extraction_prompt="Context: {text}")
    schema = GraphSchema(node_types=[], relationship_types=[], patterns=[])
    prompt = builder.build_prompt("The contract was signed.", schema)
    assert prompt.startswith("Context: The contract was signed.")
    assert "Text:" not in prompt


def test_custom_claim_prompt_with_placeholders():
    builder = SchemaPromptBuilder(
        claim_prompt="Extract claims about {entity_ids} from {text}."
    )
    prompt = builder.build_claim_prompt("Alice runs Acme.", ["person:alice", "org:acme"])
    assert "Extract claims about person:alice, org:acme from Alice runs Acme." in prompt
    assert "Known entities" not in prompt
    assert "Text:" not in prompt


def test_custom_entity_summary_prompt_with_placeholders():
    builder = SchemaPromptBuilder(
        entity_summary_prompt="Summarize {entity_name} ({entity_type}) using {descriptions}."
    )
    prompt = builder.build_entity_summary_prompt(
        descriptions=["Founded in Berlin."],
        entity_name="Acme",
        entity_type="Company",
    )
    assert "Summarize Acme (Company) using - Founded in Berlin." in prompt
    assert "Entity name:" not in prompt
    assert "Entity type:" not in prompt
    assert "Observations:" not in prompt


def test_custom_relationship_summary_prompt_with_placeholders():
    builder = SchemaPromptBuilder(
        relationship_summary_prompt="Rel {source} {rel_type} {target}: {descriptions}"
    )
    prompt = builder.build_relationship_summary_prompt(
        descriptions=["collaborated closely."],
        source="person:alice",
        target="org:acme",
        rel_type="WORKS_AT",
    )
    assert "Rel person:alice WORKS_AT org:acme:" in prompt
    assert "Relationship:" not in prompt
    assert "Observations:" not in prompt


def test_escaped_braces_stay_literal():
    builder = SchemaPromptBuilder(
        entity_summary_prompt="Literal {{entity_type}} here. Data: {descriptions}"
    )
    prompt = builder.build_entity_summary_prompt(
        descriptions=["obs one"], entity_name="Acme", entity_type="Company"
    )
    # {{entity_type}} is escaped, not a placeholder: it stays literal and the
    # fallback "Entity type:" section is still appended.
    assert "Literal {entity_type} here." in prompt
    assert "Data: - obs one" in prompt
    assert "Entity type: Company" in prompt


def test_escaped_braces_unescaped_without_placeholders():
    schema = GraphSchema(node_types=[], relationship_types=[], patterns=[])
    builder = SchemaPromptBuilder(extraction_prompt="Return {{json}} only.")
    prompt = builder.build_prompt("Some text.", schema)
    assert "Return {json} only." in prompt


def test_placeholder_like_data_is_not_expanded():
    schema = GraphSchema(node_types=[], relationship_types=[], patterns=[])
    builder = SchemaPromptBuilder(assessment_prompt="Check {text} against {existing}")
    prompt = builder.build_assessment_prompt(
        "Doc mentions {existing} literally.", schema, GraphExtraction(nodes=[], relationships=[])
    )
    assert "Doc mentions {existing} literally." in prompt
