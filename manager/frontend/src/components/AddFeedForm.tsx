import { useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { Plus } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { feedsApi } from '@/lib/api'

export function AddFeedForm() {
  const [isOpen, setIsOpen] = useState(false)
  const [rssUrl, setRssUrl] = useState('')
  const queryClient = useQueryClient()

  const createFeed = useMutation({
    mutationFn: feedsApi.create,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['feeds'] })
      setRssUrl('')
      setIsOpen(false)
    },
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (rssUrl.trim()) {
      createFeed.mutate(rssUrl.trim())
    }
  }

  if (!isOpen) {
    return (
      <Button onClick={() => setIsOpen(true)} className="mb-4">
        <Plus className="mr-2 h-4 w-4" />
        Add Podcast
      </Button>
    )
  }

  return (
    <Card className="mb-4">
      <CardHeader className="pb-3">
        <CardTitle className="text-lg">Add New Podcast</CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-3">
          <div>
            <label htmlFor="rss-url" className="block text-sm font-medium mb-1">
              RSS Feed URL
            </label>
            <input
              id="rss-url"
              type="url"
              value={rssUrl}
              onChange={(e) => setRssUrl(e.target.value)}
              placeholder="https://example.com/feed.xml"
              className="w-full px-3 py-2 border rounded-md bg-background text-foreground"
              required
              disabled={createFeed.isPending}
            />
            <p className="text-xs text-muted-foreground mt-1">
              Title, description, and artwork will be fetched automatically
            </p>
          </div>
          <div className="flex gap-2 pt-2">
            <Button type="submit" disabled={createFeed.isPending}>
              {createFeed.isPending ? 'Fetching feed...' : 'Add Feed'}
            </Button>
            <Button
              type="button"
              variant="ghost"
              onClick={() => {
                setIsOpen(false)
                setRssUrl('')
              }}
              disabled={createFeed.isPending}
            >
              Cancel
            </Button>
          </div>
          {createFeed.isError && (
            <p className="text-sm text-destructive">
              {createFeed.error.message}
            </p>
          )}
        </form>
      </CardContent>
    </Card>
  )
}
