import { useState } from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Podcast, Inbox } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { FeedView } from '@/components/FeedView'
import { EpisodeTable } from '@/components/EpisodeTable'
import { StatusOverview, type EpisodeTab } from '@/components/StatusOverview'
import { PasswordGate } from '@/components/PasswordGate'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60,
      retry: 1,
    },
  },
})

type MainTab = 'feeds' | 'episodes'

function AppContent() {
  const [mainTab, setMainTab] = useState<MainTab>('feeds')
  const [episodeTab, setEpisodeTab] = useState<EpisodeTab>('inbox')

  const handleTabClick = (tab: EpisodeTab) => {
    setEpisodeTab(tab)
    setMainTab('episodes')
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <h1 className="text-2xl font-bold">PodcastPurifier</h1>
            <nav className="flex gap-2">
              <Button
                variant={mainTab === 'feeds' ? 'default' : 'ghost'}
                onClick={() => setMainTab('feeds')}
              >
                <Podcast className="mr-2 h-4 w-4" />
                Feeds
              </Button>
              <Button
                variant={mainTab === 'episodes' ? 'default' : 'ghost'}
                onClick={() => setMainTab('episodes')}
              >
                <Inbox className="mr-2 h-4 w-4" />
                Episodes
              </Button>
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        <StatusOverview onTabClick={handleTabClick} />

        {mainTab === 'feeds' && (
          <div>
            <h2 className="text-xl font-semibold mb-4">Subscribed Podcasts</h2>
            <FeedView />
          </div>
        )}
        {mainTab === 'episodes' && (
          <div>
            <EpisodeTable activeTab={episodeTab} onTabChange={setEpisodeTab} />
          </div>
        )}
      </main>
    </div>
  )
}

function App() {
  return (
    <PasswordGate>
      <QueryClientProvider client={queryClient}>
        <AppContent />
      </QueryClientProvider>
    </PasswordGate>
  )
}

export default App
