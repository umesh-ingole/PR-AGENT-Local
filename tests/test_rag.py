import chromadb
import pytest

import rag


class FakeCollection:
    def __init__(self):
        self.documents = {}
        self.metadatas = {}
        self.deleted = []
        self.fail_upsert = False

    def get(self, include):
        assert include == ["documents", "metadatas"]
        ids = sorted(self.documents)
        return {
            "ids": ids,
            "documents": [self.documents[item] for item in ids],
            "metadatas": [self.metadatas[item] for item in ids],
        }

    def upsert(self, ids, documents, metadatas):
        if self.fail_upsert:
            raise RuntimeError("simulated Chroma failure")
        self.documents.update(zip(ids, documents))
        self.metadatas.update(zip(ids, metadatas))

    def delete(self, ids):
        self.deleted.extend(ids)
        for item in ids:
            self.documents.pop(item, None)
            self.metadatas.pop(item, None)


def test_cached_collection_refreshes_changed_and_removed_guides(
    tmp_path, monkeypatch,
):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    guide = knowledge_dir / "python_style.md"
    guide.write_text("# Python\n- Original guidance\n- Keep this rule\n", encoding="utf-8")
    (knowledge_dir / "obsolete.md").write_text(
        "# Obsolete\n- Remove this guide\n", encoding="utf-8"
    )
    collection = FakeCollection()
    client = type(
        "FakeClient",
        (),
        {"get_or_create_collection": lambda _self, _name: collection},
    )()
    monkeypatch.setattr(chromadb, "PersistentClient", lambda **_kwargs: client)
    monkeypatch.setattr(rag, "KNOWLEDGE_DIR", str(knowledge_dir))
    monkeypatch.setattr(rag, "DB_DIR", str(tmp_path / ".chroma"))
    monkeypatch.setattr(rag, "_collection_cache", None)

    first = rag._collection()
    assert first is collection
    assert collection.documents["python_style.md-0"] == "# Python\n- Original guidance"
    assert "obsolete.md-0" in collection.documents

    guide.write_text(
        "# Python\n- Updated guidance\n- Keep this rule\n- New guidance\n",
        encoding="utf-8",
    )
    (knowledge_dir / "obsolete.md").unlink()

    second = rag._collection()

    assert second is first
    assert collection.documents["python_style.md-0"] == "# Python\n- Updated guidance"
    assert collection.documents["python_style.md-1"] == "# Python\n- Keep this rule"
    assert collection.documents["python_style.md-2"] == "# Python\n- New guidance"
    assert "obsolete.md-0" not in collection.documents
    assert collection.deleted == ["obsolete.md-0"]


def test_failed_upsert_does_not_delete_existing_index_entries(
    tmp_path, monkeypatch,
):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    guide = knowledge_dir / "guide.md"
    guide.write_text("# Guide\n- Updated guidance\n", encoding="utf-8")
    collection = FakeCollection()
    collection.documents["guide.md-0"] = "# Guide\n- Old guidance"
    collection.metadatas["guide.md-0"] = {"source": "guide.md"}
    collection.documents["removed.md-0"] = "# Removed\n- Old entry"
    collection.metadatas["removed.md-0"] = {"source": "removed.md"}
    collection.fail_upsert = True
    monkeypatch.setattr(rag, "KNOWLEDGE_DIR", str(knowledge_dir))

    with pytest.raises(RuntimeError, match="simulated Chroma failure"):
        rag._sync_collection(collection)

    assert collection.documents["guide.md-0"] == "# Guide\n- Old guidance"
    assert "removed.md-0" in collection.documents
    assert collection.deleted == []


def test_empty_knowledge_directory_preserves_existing_index(
    tmp_path, monkeypatch,
):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    collection = FakeCollection()
    collection.documents["guide.md-0"] = "# Guide\n- Existing guidance"
    collection.metadatas["guide.md-0"] = {"source": "guide.md"}
    monkeypatch.setattr(rag, "KNOWLEDGE_DIR", str(knowledge_dir))

    rag._sync_collection(collection)

    assert collection.documents["guide.md-0"] == "# Guide\n- Existing guidance"
    assert collection.deleted == []
