import type { Feed, PaginatedEpisodes, EpisodeStats } from '@/types/api'

async function fetchAPI<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`/api${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
    throw new Error(error.detail || `API error: ${response.status}`)
  }

  return response.json()
}

export const feedsApi = {
  list: () => fetchAPI<Feed[]>('/feeds'),

  create: (rss_url: string) =>
    fetchAPI<Feed>(`/feeds?rss_url=${encodeURIComponent(rss_url)}`, {
      method: 'POST',
    }),

  delete: (feedId: number) =>
    fetchAPI<{ message: string; deleted_episodes: number }>(`/feeds/${feedId}`, {
      method: 'DELETE',
    }),

  updateAutoProcess: (feedId: number, autoProcess: boolean) =>
    fetchAPI<Feed>(`/feeds/${feedId}/auto-process?auto_process=${autoProcess}`, {
      method: 'PATCH',
    }),

  ingest: (feedId: number) =>
    fetchAPI<{ message: string; new_episodes: number }>(`/feeds/${feedId}/ingest`, {
      method: 'POST',
    }),
}

export interface EpisodeListParams {
  feed_id?: number
  status?: string
  exclude_statuses?: string
  page?: number
  page_size?: number
}

export const episodesApi = {
  list: (params?: EpisodeListParams) => {
    const searchParams = new URLSearchParams()
    if (params?.feed_id != null) searchParams.set('feed_id', String(params.feed_id))
    if (params?.status) searchParams.set('status', params.status)
    if (params?.exclude_statuses) searchParams.set('exclude_statuses', params.exclude_statuses)
    if (params?.page) searchParams.set('page', String(params.page))
    if (params?.page_size) searchParams.set('page_size', String(params.page_size))
    const query = searchParams.toString()
    return fetchAPI<PaginatedEpisodes>(`/episodes${query ? `?${query}` : ''}`)
  },

  queueBulk: (episodeIds: number[]) =>
    fetchAPI<{ queued: number }>('/episodes/queue', {
      method: 'POST',
      body: JSON.stringify({ episode_ids: episodeIds }),
    }),

  ignoreBulk: (episodeIds: number[]) =>
    fetchAPI<{ ignored: number }>('/episodes/ignore', {
      method: 'POST',
      body: JSON.stringify({ episode_ids: episodeIds }),
    }),

  unignoreBulk: (episodeIds: number[]) =>
    fetchAPI<{ restored: number }>('/episodes/unignore', {
      method: 'POST',
      body: JSON.stringify({ episode_ids: episodeIds }),
    }),

  unqueueBulk: (episodeIds: number[]) =>
    fetchAPI<{ unqueued: number }>('/episodes/unqueue', {
      method: 'POST',
      body: JSON.stringify({ episode_ids: episodeIds }),
    }),

  failBulk: (episodeIds: number[]) =>
    fetchAPI<{ failed: number }>('/episodes/fail', {
      method: 'POST',
      body: JSON.stringify({ episode_ids: episodeIds }),
    }),

  stats: () => fetchAPI<EpisodeStats>('/episodes/stats'),
}
