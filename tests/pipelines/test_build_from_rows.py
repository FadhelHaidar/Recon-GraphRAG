"""Tests for GraphBuilderPipeline structured-data ingestion."""

import sqlite3
from unittest.mock import AsyncMock, MagicMock

import pytest

from recon_graphrag.extraction.schema import (
    GraphSchema,
    NodeType,
    RelationshipType,
)
from recon_graphrag.extraction.structured import (
    ColumnEntity,
    RowMapping,
    RowRelationship,
    TextColumn,
)
from recon_graphrag.pipelines.graphrag_pipeline import GraphBuilderPipeline

from tests.pipelines.test_graph_builder import FakeGraphStore


ROWS = [
    {
        "order_id": "O-1",
        "supplier_name": "Acme Corp",
        "product_name": "Widget",
        "price": "9.99",
        "notes": "Widget uses the AX-9 chip made by ChipCo.",
    },
    {
        "order_id": "O-2",
        "supplier_name": "Acme Corp",
        "product_name": "Widget",
        "price": "9.99",
        "notes": "",
    },
]


@pytest.fixture
def mapping():
    return RowMapping(
        entities=[
            ColumnEntity(label="Company", identity_column="supplier_name"),
            ColumnEntity(
                label="Product",
                identity_column="product_name",
                properties={"price": "unit_price"},
                types={"unit_price": "FLOAT"},
            ),
        ],
        relationships=[RowRelationship("Company", "SUPPLIES", "Product")],
        record_id_column="order_id",
    )


@pytest.fixture
def fake_writer():
    writer = MagicMock()
    writer.write_graph_document = MagicMock(
        return_value={"documents": 1, "chunks": 2, "entities": 2, "relationships": 1}
    )
    return writer


def _make_pipeline(llm, fake_writer):
    return GraphBuilderPipeline(
        graph_store=FakeGraphStore(),
        llm=llm,
        embedder=MagicMock(),
        graph_writer=fake_writer,
        perform_entity_resolution=False,
        embed_entities=False,
        summarize_descriptions=False,
    )


@pytest.mark.asyncio
async def test_direct_only_makes_zero_llm_calls(mapping, fake_writer):
    llm = MagicMock()
    llm.ainvoke = AsyncMock()
    pipeline = _make_pipeline(llm, fake_writer)

    result = await pipeline.build_from_rows(
        [dict(row, notes="") for row in ROWS],
        mapping,
        metadata={"source": "orders"},
    )

    llm.ainvoke.assert_not_called()
    assert result["extraction"]["chunks"] == 2
    assert result["extraction"]["llm_rows"] == 0
    assert "validation" in result

    doc = fake_writer.write_graph_document.call_args.args[0]
    assert doc.document.id == "doc:orders"
    assert [chunk.metadata["record_id"] for chunk in doc.chunks] == ["O-1", "O-2"]

    entities = {e.canonical_key: e for e in doc.entities}
    assert set(entities) == {"Company:Acme Corp", "Product:Widget"}
    product = entities["Product:Widget"]
    assert product.type == "Product"
    assert product.properties["name"] == "Widget"
    assert product.properties["unit_price"] == 9.99
    assert product.properties["description"]  # deterministic, non-empty

    # Same relationship across both rows dedups to one record with a bumped count.
    (rel,) = doc.relationships
    assert rel.type == "SUPPLIES"
    assert rel.observation_count == 2


@pytest.mark.asyncio
async def test_cross_label_same_name_stays_separate(fake_writer):
    mapping = RowMapping(
        entities=[
            ColumnEntity(label="Company", identity_column="brand"),
            ColumnEntity(label="Product", identity_column="item"),
        ],
    )
    pipeline = _make_pipeline(MagicMock(), fake_writer)

    await pipeline.build_from_rows(
        [{"brand": "Mercury", "item": "Mercury"}],
        mapping,
        metadata={"source": "collide"},
        finalize=False,
    )

    doc = fake_writer.write_graph_document.call_args.args[0]
    assert len(doc.entities) == 2
    assert len({e.id for e in doc.entities}) == 2  # distinct uuids


@pytest.mark.asyncio
async def test_text_columns_merge_llm_extraction_and_anchor(mapping, fake_writer):
    mapping = RowMapping(
        entities=mapping.entities,
        relationships=mapping.relationships,
        text_columns=[TextColumn("notes", anchor_label="Product")],
        record_id_column=mapping.record_id_column,
    )
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=MagicMock(content='''
    {
        "nodes": [
            {"id": "Widget", "label": "Product", "properties": {"name": "Widget"}},
            {"id": "ChipCo", "label": "Company", "properties": {"name": "ChipCo"}}
        ],
        "relationships": []
    }
    '''))
    pipeline = _make_pipeline(llm, fake_writer)

    result = await pipeline.build_from_rows(
        ROWS,
        mapping,
        metadata={"source": "orders"},
        max_gleanings=0,
        finalize=False,
    )

    # Only the row with non-empty notes gets an LLM pass.
    assert llm.ainvoke.await_count == 1
    assert result["extraction"]["llm_rows"] == 1

    doc = fake_writer.write_graph_document.call_args.args[0]
    entities = {e.canonical_key for e in doc.entities}
    # Direct scoped keys plus LLM name-keyed entities (merged later by resolution).
    assert entities == {"Company:Acme Corp", "Product:Widget", "Widget", "ChipCo"}

    mentions = [r for r in doc.relationships if r.type == "MENTIONS"]
    # Anchor edge to ChipCo; the self-link to "Widget" is skipped.
    assert len(mentions) == 1
    (mention,) = mentions
    assert mention.properties["source_canonical_key"] == "Product:Widget"
    assert mention.properties["target_canonical_key"] == "ChipCo"


@pytest.mark.asyncio
async def test_custom_extraction_schema_constrains_llm_pass(mapping, fake_writer):
    mapping = RowMapping(
        entities=mapping.entities,
        text_columns=[TextColumn("notes", anchor_label="Product")],
    )
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=MagicMock(content='''
    {
        "nodes": [
            {"id": "AX-9", "label": "Component", "properties": {"name": "AX-9"}},
            {"id": "ChipCo", "label": "Company", "properties": {"name": "ChipCo"}}
        ],
        "relationships": []
    }
    '''))
    pipeline = _make_pipeline(llm, fake_writer)

    companies_only = GraphSchema(
        node_types=[NodeType(label="Company")],
        relationship_types=[RelationshipType(label="MENTIONS")],
    )
    await pipeline.build_from_rows(
        [ROWS[0]],
        mapping,
        metadata={"source": "orders"},
        extraction_schema=companies_only,
        max_gleanings=0,
        finalize=False,
    )

    doc = fake_writer.write_graph_document.call_args.args[0]
    keys = {e.canonical_key for e in doc.entities}
    # Component "AX-9" was dropped by the validator against the custom schema.
    assert "AX-9" not in keys
    assert "ChipCo" in keys


@pytest.mark.asyncio
async def test_build_from_csv_and_sql(tmp_path, mapping, fake_writer):
    pipeline = _make_pipeline(MagicMock(), fake_writer)

    path = tmp_path / "orders.csv"
    path.write_text(
        "order_id,supplier_name,product_name,price\nO-1,Acme Corp,Widget,9.99\n",
        encoding="utf-8",
    )
    result = await pipeline.build_from_csv(path, mapping, finalize=False)
    assert result["extraction"]["document_id"] == "doc:orders-csv"
    assert result["extraction"]["chunks"] == 1

    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE orders (order_id TEXT, supplier_name TEXT, product_name TEXT, price REAL)"
    )
    connection.execute("INSERT INTO orders VALUES ('O-1', 'Acme Corp', 'Widget', 9.99)")
    result = await pipeline.build_from_sql(
        connection, "SELECT * FROM orders", mapping, finalize=False
    )
    assert result["extraction"]["chunks"] == 1
    connection.close()

    doc = fake_writer.write_graph_document.call_args.args[0]
    product = next(e for e in doc.entities if e.canonical_key == "Product:Widget")
    assert product.properties["unit_price"] == 9.99  # already-typed value passes through


@pytest.mark.asyncio
async def test_invalid_mapping_raises_before_any_work(mapping, fake_writer):
    pipeline = _make_pipeline(MagicMock(), fake_writer)
    bad = RowMapping(
        entities=[ColumnEntity(label="P", identity_column="a")],
        relationships=[RowRelationship("P", "REL", "Missing")],
    )
    with pytest.raises(ValueError, match="unknown label"):
        await pipeline.build_from_rows([{"a": "x"}], bad)
    fake_writer.write_graph_document.assert_not_called()
