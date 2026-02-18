"""
IMI Notebook Router
====================
CRUD + execute endpoints for interactive IMI notebooks.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from imi_notebook import (
    create_notebook,
    get_notebook,
    list_notebooks,
    add_cell,
    execute_cell,
    delete_notebook,
)

router = APIRouter(prefix="/klaus/imi", tags=["imi-notebook"])


class CreateNotebookRequest(BaseModel):
    title: str


class AddCellRequest(BaseModel):
    type: str
    source: str


@router.post("/notebook/create")
async def api_create_notebook(req: CreateNotebookRequest):
    return create_notebook(req.title)


@router.get("/notebook/list")
async def api_list_notebooks():
    return list_notebooks()


@router.get("/notebook/{notebook_id}")
async def api_get_notebook(notebook_id: str):
    nb = get_notebook(notebook_id)
    if nb is None:
        raise HTTPException(status_code=404, detail="Notebook not found")
    return nb


@router.post("/notebook/{notebook_id}/cell")
async def api_add_cell(notebook_id: str, req: AddCellRequest):
    try:
        cell = add_cell(notebook_id, req.type, req.source)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if cell is None:
        raise HTTPException(status_code=404, detail="Notebook not found")
    return cell


@router.post("/notebook/{notebook_id}/cell/{cell_id}/execute")
async def api_execute_cell(notebook_id: str, cell_id: str):
    cell = execute_cell(notebook_id, cell_id)
    if cell is None:
        raise HTTPException(status_code=404, detail="Notebook or cell not found")
    return cell


@router.delete("/notebook/{notebook_id}")
async def api_delete_notebook(notebook_id: str):
    if not delete_notebook(notebook_id):
        raise HTTPException(status_code=404, detail="Notebook not found")
    return {"status": "deleted", "id": notebook_id}
