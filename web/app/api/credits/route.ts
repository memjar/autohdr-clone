/**
 * Credits API Routes
 * Handles credit purchases (top-ups) and credit history
 */

import { NextRequest, NextResponse } from 'next/server';
import { CreditLog } from '@/lib/types';

// In-memory stores
const creditLogs = new Map<string, CreditLog[]>();

const DEMO_USER_ID = 'demo-user';

// Credit pack pricing
const CREDIT_PACKS = {
  small: { credits: 50, price: 30 },      // $0.60/photo
  medium: { credits: 200, price: 100 },   // $0.50/photo
  large: { credits: 500, price: 225 },    // $0.45/photo
  xl: { credits: 1000, price: 400 },      // $0.40/photo
} as const;

// GET /api/credits - Get credit history
export async function GET(req: NextRequest) {
  const userId = req.headers.get('x-user-id') || DEMO_USER_ID;
  const url = new URL(req.url);
  const limit = parseInt(url.searchParams.get('limit') || '20');
  const offset = parseInt(url.searchParams.get('offset') || '0');

  const userLogs = creditLogs.get(userId) || [];

  // Sort by date descending and paginate
  const sortedLogs = [...userLogs]
    .sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime())
    .slice(offset, offset + limit);

  return NextResponse.json({
    success: true,
    data: {
      logs: sortedLogs,
      total: userLogs.length,
      limit,
      offset,
    },
  });
}

// POST /api/credits - Purchase credit pack (top-up)
export async function POST(req: NextRequest) {
  const userId = req.headers.get('x-user-id') || DEMO_USER_ID;

  try {
    const body = await req.json();
    const { packId } = body as { packId: keyof typeof CREDIT_PACKS };

    if (!packId || !CREDIT_PACKS[packId]) {
      return NextResponse.json(
        {
          success: false,
          error: 'Invalid pack ID',
          availablePacks: Object.entries(CREDIT_PACKS).map(([id, pack]) => ({
            id,
            credits: pack.credits,
            price: pack.price,
            pricePerPhoto: (pack.price / pack.credits).toFixed(2),
          })),
        },
        { status: 400 }
      );
    }

    const pack = CREDIT_PACKS[packId];

    // In production, this would integrate with Stripe
    // For now, simulate successful purchase

    // Get current balance (would come from subscription in real app)
    const userLogs = creditLogs.get(userId) || [];
    const lastLog = userLogs[userLogs.length - 1];
    const previousBalance = lastLog?.newBalance || 10; // Default 10 for free tier

    const newLog: CreditLog = {
      id: `log_${Date.now()}`,
      subscriptionId: `sub_${userId}`,
      action: 'purchase',
      amount: pack.credits,
      newBalance: previousBalance + pack.credits,
      description: `Purchased ${packId} credit pack (${pack.credits} credits for $${pack.price})`,
      createdAt: new Date(),
    };

    userLogs.push(newLog);
    creditLogs.set(userId, userLogs);

    return NextResponse.json({
      success: true,
      data: {
        creditsAdded: pack.credits,
        amountCharged: pack.price,
        newBalance: newLog.newBalance,
        log: newLog,
      },
      message: `Successfully added ${pack.credits} credits to your account!`,
    });
  } catch (error) {
    console.error('Credit purchase error:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to purchase credits' },
      { status: 500 }
    );
  }
}

// GET /api/credits/packs - Get available credit packs
export async function OPTIONS(req: NextRequest) {
  return NextResponse.json({
    success: true,
    data: {
      packs: Object.entries(CREDIT_PACKS).map(([id, pack]) => ({
        id,
        credits: pack.credits,
        price: pack.price,
        pricePerPhoto: `$${(pack.price / pack.credits).toFixed(2)}`,
        savings: id === 'small' ? null : `${Math.round((1 - (pack.price / pack.credits) / 0.6) * 100)}% off`,
      })),
    },
  });
}
