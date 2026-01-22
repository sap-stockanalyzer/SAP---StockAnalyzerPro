AION Bots Page (Stacked layout matching the sketch render)

Files
- page.tsx  -> THIS is the actual Next.js App Router route file (drop into your route folder).

Why there used to be two files:
- The previous zip accidentally included an older "cards/tabs" page.tsx plus the newer stacked layout as bots_page.tsx.
- This zip removes that confusion: page.tsx is the stacked version.

Install
1) Copy page.tsx into your route folder, e.g.
   app/admin/bots/page.tsx
2) Ensure your existing UI has the same shared components (shadcn/ui) already used in your project.

Notes
- The toggle is a persistent ON/OFF button (red/green) per bot.
- Save applies bot rules; Reset restores defaults.
