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

O backend precisa de Python 3.11+ instalado. Primeiro confirme:

```powershell
python --version
```

Pelo diretorio raiz do projeto, a forma recomendada e:

```powershell
.\scripts\run-backend.ps1
```

Se o PowerShell bloquear a execucao de scripts:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run-backend.ps1
```

Execucao manual, sem depender de `Activate.ps1`:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Se a `.venv` ja existir e falhar com `did not find executable`, ela esta
apontando para um Python removido. Recrie:

```powershell
cd backend
Remove-Item .venv -Recurse -Force
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## Testes

```powershell
cd backend
.\.venv\Scripts\python.exe -m unittest discover tests
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
