Sim1Shot2D — 2D Single‑Shot Seismic Modeling & Imaging
======================================================

Sim1Shot2D is an educational and research‑oriented tool for:

- 2D acoustic finite‑difference single‑shot forward modeling (First-Order Acoustic)
- working with velocity models loaded from SEG‑Y or simple layered models
- configuring source and receiver geometry
- saving snapshots and seismograms
- reverse‑time migration (RTM) imaging and basic post‑processing

The GUI is built with PyQt5 and provides:

- interactive visualization of the velocity model and snapshots
- seismogram display and basic amplitude scaling
- project files for saving and restoring modeling setups

Installation & Run
------------------

1. Clone the repository:

   ```bash
   git clone https://github.com/sergeevsn/sim1shot2d.git
   cd sim1shot2d/gui
   ```

2. Create a virtual environment (recommended) and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   # or:  .venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   ```

3. Start the application:

   ```bash
   python main.py
   ```

   You can try different setup project files in ```projects/``` directory

Author & Contacts
-----------------

Sim1Shot2D is developed and maintained by **Sergey Sergeev**.

- Web site: `https://sergeevsergei.ru/`
- Telegram: `https://t.me/twowaytime`

For questions, feedback, or suggestions feel free to contact the author via the web site or Telegram channel.

