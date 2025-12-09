import { useQuery } from '@tanstack/react-query'
import { feedsApi } from '@/lib/api'
import { FeedCard } from './FeedCard'
import { AddFeedForm } from './AddFeedForm'

export function FeedView() {
  const { data: feeds, isLoading, error } = useQuery({
    queryKey: ['feeds'],
    queryFn: feedsApi.list,
  })

  if (isLoading) {
    return (
      <div>
        <AddFeedForm />
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[1, 2, 3].map((i) => (
            <div
              key={i}
              className="h-32 rounded-xl border bg-card animate-pulse"
            />
          ))}
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div>
        <AddFeedForm />
        <div className="text-center py-8 text-destructive">
          Failed to load feeds: {error.message}
        </div>
      </div>
    )
  }

  return (
    <div>
      <AddFeedForm />
      {!feeds?.length ? (
        <div className="text-center py-8 text-muted-foreground">
          No feeds yet. Add your first podcast feed to get started.
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {feeds.map((feed) => (
            <FeedCard key={feed.id} feed={feed} />
          ))}
        </div>
      )}
    </div>
  )
}
