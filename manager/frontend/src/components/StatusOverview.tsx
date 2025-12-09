import { useQuery } from '@tanstack/react-query'
import { CheckCircle, Clock, Loader2, AlertCircle, Search } from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'
import { episodesApi } from '@/lib/api'

export type EpisodeTab = 'inbox' | 'queued' | 'cleaned' | 'ignored' | 'failed'

interface StatusOverviewProps {
  onTabClick: (tab: EpisodeTab) => void
}

export function StatusOverview({ onTabClick }: StatusOverviewProps) {
  const { data: stats } = useQuery({
    queryKey: ['episode-stats'],
    queryFn: episodesApi.stats,
    refetchInterval: 5000,
  })

  if (!stats) return null

  const items: Array<{
    label: string
    count: number
    icon: typeof Search
    color: string
    tab: EpisodeTab
  }> = [
    {
      label: 'Inbox',
      count: stats.discovered + stats.failed,
      icon: Search,
      color: 'text-blue-500',
      tab: 'inbox',
    },
    {
      label: 'Queued',
      count: stats.queued + stats.processing,
      icon: Clock,
      color: 'text-yellow-500',
      tab: 'queued',
    },
    {
      label: 'Cleaned',
      count: stats.cleaned,
      icon: CheckCircle,
      color: 'text-green-500',
      tab: 'cleaned',
    },
    {
      label: 'Failed',
      count: stats.failed,
      icon: AlertCircle,
      color: 'text-red-500',
      tab: 'failed',
    },
  ]

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
      {items.map((item) => (
        <Card
          key={item.tab}
          className="cursor-pointer hover:bg-accent transition-colors"
          onClick={() => onTabClick(item.tab)}
        >
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <item.icon className={`h-5 w-5 ${item.color}`} />
              <div>
                <div className="text-2xl font-bold">{item.count}</div>
                <div className="text-sm text-muted-foreground">{item.label}</div>
              </div>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}
