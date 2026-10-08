create table if not exists public.latest_entries (
  user_id uuid primary key references auth.users (id) on delete cascade,
  latest_entry jsonb not null,
  updated_at timestamptz not null default now()
);

alter table public.latest_entries enable row level security;

drop policy if exists "Users can read their own latest entry"
  on public.latest_entries;
create policy "Users can read their own latest entry"
  on public.latest_entries
  for select
  to authenticated
  using (auth.uid() = user_id);

drop policy if exists "Users can insert their own latest entry"
  on public.latest_entries;
create policy "Users can insert their own latest entry"
  on public.latest_entries
  for insert
  to authenticated
  with check (auth.uid() = user_id);

drop policy if exists "Users can update their own latest entry"
  on public.latest_entries;
create policy "Users can update their own latest entry"
  on public.latest_entries
  for update
  to authenticated
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

grant select, insert, update on public.latest_entries to authenticated;
