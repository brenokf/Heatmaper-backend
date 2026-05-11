# Wi-Fi Heatmap Detection API

Backend FastAPI opcional para detectar pontos visuais em uma planta baixa. A geracao do heatmap fica no frontend Next.js.

## Coordenadas

- A planta esta em pixels.
- Origem: `(0,0)` no canto superior esquerdo da planta.
- Eixo `X`: cresce para a direita.
- Eixo `Y`: cresce para baixo.
- O eixo `Y` nao e invertido em nenhuma etapa.
- Escala padrao: `1 metro = 50 px`, equivalente a `2 cm/px`.

## Rodar

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Testes

```powershell
cd backend
.venv\Scripts\python.exe -m unittest discover tests
```

## Endpoint

`POST /api/detect-points`

Campos `multipart/form-data`:

- `floor_plan`: imagem da planta

Retorna pontos detectados:

- `points`
- `markers_detected`
- `width`
- `height`
